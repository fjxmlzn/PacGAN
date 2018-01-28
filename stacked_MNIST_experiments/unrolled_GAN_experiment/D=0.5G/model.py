from __future__ import division
import os, csv
import time, pickle
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

from math import log, ceil

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
  
# from https://www.zealseeker.com/archives/jensen-shannon-divergence-jsd-python/
class JSD:
    def KLD(self,p,q):
        if 0 in q :
            raise ValueError
        return sum(_p * log(_p/_q) for (_p,_q) in zip(p,q) if _p!=0)

    def JSD_core(self,p,q):
        M = [0.5*(_p+_q) for _p,_q in zip(p,q)]
        return 0.5*self.KLD(p,M)+0.5*self.KLD(q,M)

class DCGAN(object):
    def __init__(self, sess, test_samples, input_height=108, input_width=108, batch_size=64, sample_num=64, output_height=64, output_width=64, z_dim=100, c_dim=3, dataset_name='default', checkpoint_dir=None, sample_dir=None, packing_num=1, num_training_sample=25600, num_test_sample=26000):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.packing_num = packing_num
        self.num_training_sample = num_training_sample
        self.num_test_sample = num_test_sample
        self.test_samples = test_samples
        self.sample_dir = sample_dir
        
        self.sess = sess

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        self.load_mnist()
        self.c_dim = 3 * self.packing_num

        self.grayscale = (self.c_dim == 1)

        self.build_model()
        
        from evaluate import evaluate
        self.evaluate = evaluate

    def build_model(self):
        self.y = None

        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs
    
        self.z = []
        for i in range(self.packing_num):
            self.z.append(tf.placeholder(
                tf.float32, [None, self.z_dim], name='z{}'.format(i)))
        self.z_sum = histogram_summary("z", tf.concat(self.z, 0))
    
        self.G_sep = []
        for i in range(self.packing_num):
            self.G_sep.append(self.generator(self.z[i], self.y))
        self.G                  = tf.concat(self.G_sep, 3)
        self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
        self.sampler            = self.sampler(self.z[0], self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", tf.concat(self.G_sep, 0))

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        print("Number of parameters in discriminator: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in self.d_vars])))
        print("Number of parameters in generator: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in self.g_vars])))

        self.saver = tf.train.Saver()

    def train(self, config):
        if config.log_metrics:
            self.metric_path = os.path.join(config.work_dir, "metrics.csv")
            self.field_names = ["epoch", "idx", "mode coverage", "KL", "details"]
            with open(self.metric_path, "wb") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
                writer.writeheader()
  
        d_optim = tf.train.AdamOptimizer(config.disc_learning_rate, beta1=config.beta1, beta2=config.beta2) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.gen_learning_rate, beta1=config.beta1, beta2=config.beta2) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        sample_z = np.random.normal(size=(self.sample_num , self.z_dim))
    
        sample_inputs = self.generate_stacked_mnist(self.sample_num)
  
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            batch_idxs = config.num_training_sample // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images = self.generate_stacked_mnist(config.batch_size)
            
                feed_dict_z = {}
                for i in range(self.packing_num):
                    feed_dict_z[self.z[i]] = np.random.normal(size=(config.batch_size, self.z_dim)).astype(np.float32)
                
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict=dict({ 
                        self.inputs: batch_images, 
                        #self.z: batch_z 
                    }, **feed_dict_z))
                #self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict=dict({ 
                        #self.z: batch_z 
                    }, **feed_dict_z))
                #self.writer.add_summary(summary_str, counter)
              
                errD_fake = self.d_loss_fake.eval(dict({ 
                    #self.z: batch_z 
                }, **feed_dict_z))
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval(dict({
                    #self.z: batch_z
                }, **feed_dict_z))

                counter += 1
                print("Epoch: [%2d] [%8d/%8d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 2000) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict=dict({
                            #self.z: sample_z,
                            self.inputs: sample_inputs,
                        }, **feed_dict_z))
                    save_images(samples[0 : 64], image_manifold_size(samples[0 : 64].shape[0]),
                        '{}/train_{:02d}_{:08d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                    if config.log_metrics:
                        self.log_metrics(epoch, idx)

                if np.mod(counter, 2000) == 2:
                    self.save(config.checkpoint_dir, counter)
            if config.log_metrics:
                self.log_metrics(epoch, -1)

          
    def log_metrics(self, epoch, idx):
        results = []
        for i in range(int(ceil(float(self.num_test_sample) / float(self.batch_size)))):
            input_z_samples = self.test_samples[i * self.batch_size : (i + 1) * self.batch_size]
            samples = self.sess.run(self.sampler, feed_dict={self.z[0]: input_z_samples})
            
            samples = (samples + 1.0) / 2.0
            
            dect0 = self.evaluate(np.reshape(samples[:, :, :, 0], (self.batch_size, self.output_height, self.output_width, 1)))
            dect1 = self.evaluate(np.reshape(samples[:, :, :, 1], (self.batch_size, self.output_height, self.output_width, 1)))
            dect2 = self.evaluate(np.reshape(samples[:, :, :, 2], (self.batch_size, self.output_height, self.output_width, 1)))
            
            new_results = zip(dect0, dect1, dect2)
            if len(results) + len(new_results) > self.num_test_sample:
                results.extend(new_results[0 : (self.num_test_sample - len(results))])
            else:
                results.extend(new_results)
            
            if i % 10 == 0:
                save_images(np.reshape(samples[0, :, :, 0], (1, self.output_height, self.output_width, 1)), image_manifold_size(1), os.path.join(self.sample_dir, "eva_epoch{}_idx{}_i{}_dect{}.png".format(epoch, idx, i, dect0[0])))

        map = {}
        for result in results:
            if result in map:
                map[result] += 1
            else:
                map[result] = 1
        num_mode = len(map.keys())
        p = np.zeros(1000)
        p[0:num_mode] = map.values()
        p = p / np.sum(p)
        q = [1.0 / 1000.0] * 1000
        kl = JSD().KLD(p, q) 

        with open(self.metric_path, "ab") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            writer.writerow({
                "epoch": epoch,
                "idx": idx,
                "mode coverage": num_mode, 
                "KL": kl,
                "details": map
            })

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(self.d_bn1(conv2d(image, 4, name='d_h0_conv', k_h=3, k_w=3)), leak=0.3)
            h1 = lrelu(self.d_bn2(conv2d(h0, 8, name='d_h1_conv', k_h=3, k_w=3)), leak=0.3)
            h2 = lrelu(self.d_bn3(conv2d(h1, 16, name='d_h2_conv', k_h=3, k_w=3)), leak=0.3)
            h4 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4                

    def generator(self, z, y=None):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, 64 * 4 * 4, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, 4, 4, 64])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, 7, 7, 32], name='g_h1', with_w=True, k_h=3, k_w=3)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, 14, 14, 16], name='g_h2', with_w=True, k_h=3, k_w=3)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, 28, 28, 8], name='g_h3', with_w=True, k_h=3, k_w=3)
            h3 = tf.nn.relu(self.g_bn3(h3))
            
            h4 = conv2d(h3, 3, d_h=1, d_w=1, k_h=3, k_w=3, name='g_h4')

            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, 64 * 4 * 4, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, 4, 4, 64])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, 7, 7, 32], name='g_h1', with_w=True, k_h=3, k_w=3)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, 14, 14, 16], name='g_h2', with_w=True, k_h=3, k_w=3)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, 28, 28, 8], name='g_h3', with_w=True, k_h=3, k_w=3)
            h3 = tf.nn.relu(self.g_bn3(h3))
            
            h4 = conv2d(h3, 3, d_h=1, d_w=1, k_h=3, k_w=3, name='g_h4')

            return tf.nn.tanh(h4)

    def generate_stacked_mnist(self, num_sample):
        ids = np.random.randint(0, self.mnist_X.shape[0], size=(num_sample, 3 * self.packing_num))
        X_training = np.zeros(shape=(ids.shape[0], 28, 28, ids.shape[1]))
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                X_training[i, :, :, j] = self.mnist_X[ids[i, j], :, :, 0]
        
        return X_training / 255.0 * 2.0 - 1.0 # unrolled GAN use input data in [-1, 1]    
        
    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
    
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        self.mnist_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)
      
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
