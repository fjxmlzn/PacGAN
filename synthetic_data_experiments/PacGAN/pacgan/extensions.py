import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot, rc

import os, theano, numpy, imageio, glob, re, csv, scipy, math
from blocks.serialization import dump
from blocks.extensions import SimpleExtension
from theano.sandbox.rng_mrg import MRG_RandomStreams
from datetime import datetime
from math import log

import cPickle as pickle

from pacgan.distributions import circle_gaussian_mixture

import logging
logger = logging.getLogger(__name__)

# from https://www.zealseeker.com/archives/jensen-shannon-divergence-jsd-python/
class JSD:
    def KLD(self,p,q):
        if 0 in q :
            raise ValueError
        return sum(_p * log(_p/_q) for (_p,_q) in zip(p,q) if _p!=0)

    def JSD_core(self,p,q):
        M = [0.5*(_p+_q) for _p,_q in zip(p,q)]
        return 0.5*self.KLD(p,M)+0.5*self.KLD(q,M)

class ModelLogger(SimpleExtension):
    def __init__(self, folder, file_prefix="model", use_cpickle=True, **kwargs):
        super(ModelLogger, self).__init__(**kwargs)
        self.file_prefix = file_prefix
        self.folder = folder
        self.use_cpickle = use_cpickle
        self.counter = 0

    def do(self, callback_name, *args):
        """Pickle the model to the disk.
        """
        logger.info("ModelLogger has started")
        path = os.path.join(self.folder, "{}{}.tar".format(self.file_prefix, self.counter))
        with open(path, 'wb') as f:
            dump(self.main_loop.model, f, use_cpickle=self.use_cpickle)
        logger.info("ModelLogger has finished")
        self.counter += 1
        
class GraphLogger(SimpleExtension):
    def __init__(self, num_modes, num_samples, dimension, r, std, folder, theano_rng=None, figure_file_prefix="figure", points_file_prefix="points", use_cpickle=True, only_pkl=False, **kwargs):
        super(GraphLogger, self).__init__(**kwargs)
        
        if theano_rng is None:
            theano_rng = MRG_RandomStreams(2333)
        self.z = circle_gaussian_mixture(num_modes=num_modes, num_samples=num_samples, dimension=dimension, r=r, std=std, theano_rng=theano_rng).eval()
        
        self.figure_file_prefix = figure_file_prefix
        self.points_file_prefix = points_file_prefix
        self.folder = folder
        self.use_cpickle = use_cpickle
        self.counter = 0   
        self.only_pkl = only_pkl
        
    def do(self, callback_name, *args):
        """
        Draw the generated x to the disk.
        """
        logger.info("GraphLogger has started")
        if callback_name == "after_epoch":
            path = os.path.join(self.folder, "{}{}.png".format(self.figure_file_prefix, self.counter))
            
            gan, = self.main_loop.model.top_bricks
            gan_x_tilde = gan.decoder.apply(self.z)
            samples = theano.function([], [gan_x_tilde])()
            gan_x_tilde = samples[0]
            
            pkl_path = os.path.join(self.folder, "{}{}.pkl".format(self.points_file_prefix, self.counter))
            with open(pkl_path, "wb") as f:
                pickle.dump(gan_x_tilde, f)
            
            if not self.only_pkl:
                fig = pyplot.figure(self.counter + 1)
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                
                ax.set_xlim([-6, 6])
                ax.set_ylim([-6, 6])
                ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
                ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
                
                '''
                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_xticks([-2, -1, 0, 1, 2])
                ax.set_yticks([-2, -1, 0, 1, 2])
                '''
                ax.set_xlabel('$x_1$')
                ax.set_ylabel('$x_2$')
                ax.scatter(gan_x_tilde[:, 0], gan_x_tilde[:, 1], marker='.', c='black', alpha=0.3)
                pyplot.tight_layout()
                pyplot.savefig(path, transparent=True, bbox_inches='tight')
                fig.clear()
            self.counter += 1
        elif callback_name == "after_training":
            if not self.only_pkl:
                image_list = glob.glob(os.path.join(self.folder, "*.png"))
                image_list = sorted(image_list, key=lambda x: int(re.match(".+{}([0-9]+).png".format(self.figure_file_prefix), x).groups()[0]))
                frames = []
                for image_file in image_list:
                    frames.append(imageio.imread(image_file))
                imageio.mimsave(os.path.join(self.folder, "{}.gif".format(self.figure_file_prefix)), frames, "GIF", duration=0.1)
        
        logger.info("GraphLogger has finished")
        
class MetricLogger(SimpleExtension):
    def __init__(self, means, variances, folder, points_file_prefix="points", metric_file_prefix="metrics", dis_thre=3, **kwargs):
        ''' 
        Only consider the diagonal of variance matrices!!!
        '''
        super(MetricLogger, self).__init__(**kwargs)
        self.means = means
        self.variances = variances
        self.folder = folder
        self.points_file_prefix = points_file_prefix
        self.metric_file_prefix = metric_file_prefix
        self.counter = 0
        self.field_names = ["epoch", "loglikelihood", "mode coverage", "good points percentage", "variance of coverage", "entropy of coverage", "JSD", "detail of coverage"]
        self.dis_thre = dis_thre
        
        self.norms = []
        for i in range(len(self.means)):
            self.norms.append(scipy.stats.norm(loc = self.means[i], scale = numpy.diag(self.variances[i])))
        
        with open(os.path.join(self.folder, "{}.csv".format(self.metric_file_prefix)), "ab") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = self.field_names)
            writer.writeheader()
        
    def do(self, callback_name, *args):
        #print(datetime.now().strftime("%H:%M:%S.%f"))
        pkl_path = os.path.join(self.folder, "{}{}.pkl".format(self.points_file_prefix, self.counter))
        with open(pkl_path, "rb") as f:
            gan_x_tilde = pickle.load(f)
        #print(datetime.now().strftime("%H:%M:%S.%f"))
            
        # calculate loglikelihood
        loglikelihood = 0
        for i in range(gan_x_tilde.shape[0]):
            pos = 0
            for j in range(len(self.means)):
                pos += numpy.prod(self.norms[j].pdf(gan_x_tilde[i, :]))
            pos += 1e-200
            loglikelihood += numpy.log(pos)   
        #print(datetime.now().strftime("%H:%M:%S.%f"))
        
        # map points to modes
        detail_of_coverage = numpy.zeros(len(self.means))
        for i in range(gan_x_tilde.shape[0]):
            min_dis = 10000000
            min_dis_j = -1
            for j in range(len(self.means)):
                dis_vector = numpy.array(gan_x_tilde[i, :], dtype = float) - numpy.array(self.means[j], dtype = float)
                variance_vector = numpy.array(numpy.diag(self.variances[j]), dtype = float)
                variance_vector[variance_vector == 0] = variance_vector[variance_vector != 0][1] # ugly fix for 1200D
                dis = math.sqrt(numpy.sum((dis_vector * dis_vector) / variance_vector))
                if dis < min_dis:
                    min_dis = dis
                    min_dis_j = j
            if min_dis < self.dis_thre:
                detail_of_coverage[min_dis_j] += 1
        #print(datetime.now().strftime("%H:%M:%S.%f"))
         
        mode_coverage = float(len(numpy.nonzero(detail_of_coverage)[0]))
        good_points_percentage = float(numpy.sum(detail_of_coverage)) / float(gan_x_tilde.shape[0])
        variance_of_coverage = detail_of_coverage.var()
        entropy_of_coverage = scipy.stats.entropy(detail_of_coverage / numpy.sum(detail_of_coverage))
        jsd = JSD().JSD_core(detail_of_coverage / numpy.sum(detail_of_coverage), [1.0 / len(self.means)] * len(self.means))
        
        #print(datetime.now().strftime("%H:%M:%S.%f"))
         
        # output results
        with open(os.path.join(self.folder, "{}.csv".format(self.metric_file_prefix)), "ab") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            writer.writerow({
                "epoch": self.counter,  
                "loglikelihood": loglikelihood,
                "mode coverage": mode_coverage,
                "good points percentage": good_points_percentage, 
                "variance of coverage": variance_of_coverage,
                "entropy of coverage": entropy_of_coverage,
                "JSD": jsd,
                "detail of coverage": detail_of_coverage
            })    
        #print(datetime.now().strftime("%H:%M:%S.%f"))
        
        self.counter += 1