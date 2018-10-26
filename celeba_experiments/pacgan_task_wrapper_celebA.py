"""
Code based on https://github.com/carpedm20/DCGAN-tensorflow
and https://github.com/igul222/improved_wgan_training
"""

import os, random, time, math, pickle, csv, re
from glob import glob
import numpy as np
from six.moves import xrange
import heapq as hq
import copy

from ops import *
from utils import *

import tensorflow as tf

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class PacGANTaskWrapper:
    def __init__(self, _work_dir, _config):
        self._work_dir = _work_dir
        self._config = _config

    def main(self):
        seed = random.randint(1, 100000)
        path = os.path.join(self._work_dir, ".numpyseed")
        f = open(path, "w")
        f.write("{}\n".format(seed))
        f.close()
        np.random.seed(seed)
        
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:
            self.packing_num = self._config["packing_num"]
        
            self.sess = sess
            self.batch_size = 64
            self.sample_num = 64
            
            self.input_height=108
            self.input_width=108
            self.output_height = 64
            self.output_width = 64
            
            self.z_dim = 100
            
            self.gf_dim = 64
            self.df_dim = 64
            
            self.d_bn1 = batch_norm(name="d_bn1")
            self.d_bn2 = batch_norm(name="d_bn2")
            self.d_bn3 = batch_norm(name="d_bn3")
            
            self.g_bn0 = batch_norm(name='g_bn0')
            self.g_bn1 = batch_norm(name='g_bn1')
            self.g_bn2 = batch_norm(name='g_bn2')
            self.g_bn3 = batch_norm(name='g_bn3')
            
            self.checkpoint_dir = os.path.join(self._work_dir, "checkpoint")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            self.sample_dir = os.path.join(self._work_dir, "samples")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
            self.data_dir = "celebA" #"cifar-10-batches-py"
            self.metric_path = os.path.join(self._work_dir, "metrics.csv")
            
            self.learning_rate = 0.0002
            self.epoch = 25
            self.beta1 = 0.5
            
            self.c_dim = 3
            self.data = glob(os.path.join("celebA/img_align_celeba", "*.jpg"))
            
            self.inception_score_sample_num = 50000
            
            self.build_model()
            self.train()
            nbatch, topK, runs = 16, 20, 1
            self.get_collisions(nbatch, topK, runs)
   
    def build_model(self):
        image_dims = [self.output_height, self.output_width, self.c_dim * self.packing_num]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        
        self.z = []
        for i in range(self.packing_num):
            self.z.append(tf.placeholder(
                tf.float32, [None, self.z_dim], name='z{}'.format(i)))
                
        self.G_sep = []
        for i in range(self.packing_num):
            self.G_sep.append(self.generator(self.z[i])) 
        self.G                  = tf.concat(self.G_sep, 3)
        
        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        self.sampler = self.sampler(self.z[0])
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        
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

        self.d_loss = self.d_loss_real + self.d_loss_fake
        
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        print("Number of parameters in discriminator: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in self.d_vars])))
        print("Number of parameters in generator: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in self.g_vars])))

        self.saver = tf.train.Saver()
        
    def generator(self, z):
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE) as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
        
    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
        
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
        
    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4
                
    def train(self):
        self.field_names = ["epoch", "idx", "inception score", "inception score std"]
        with open(self.metric_path, "wb") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = self.field_names)
            writer.writeheader()
    
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)    

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()      

        
        sample_feed_dict_z = {}
        for i in range(self.packing_num):
            sample_feed_dict_z[self.z[i]] = np.random.normal(size = (self.batch_size, self.z_dim)).astype(np.float32)  
        #sample_inputs = self.data[0 : self.batch_size]
        
        counter = 0
        start_time = time.time()
        
        for epoch in xrange(self.epoch):
            ## random packing and random ordering of real data
            ids = np.reshape(np.arange(0,len(self.data)),(len(self.data),1))
            np.random.shuffle(ids)
            print ids.shape
            for temp in range(self.packing_num-1):
                ids_pack = np.reshape(np.arange(0,len(self.data)),(len(self.data),1))
                np.random.shuffle(ids_pack)
                ids = np.concatenate((ids,ids_pack), axis=1)
            batch_idxs = len(self.data) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_files = []
                for index in xrange(idx * self.batch_size, (idx + 1) * self.batch_size):
                    batch_files.append([self.data[pack_index] for pack_index in ids[index]])
                batch = []
                for pack_files in batch_files:
                    batch.append(np.concatenate([get_image(batch_file,
                            input_height=self.input_height,
                            input_width=self.input_width,
                            resize_height=self.output_height,
                            resize_width=self.output_width,
                            crop=True,
                            grayscale=False) for batch_file in pack_files], axis=2))
                batch_images = np.array(batch).astype(np.float32)
                feed_dict_z = {}
                for i in range(self.packing_num):
                    feed_dict_z[self.z[i]] = np.random.normal(size = (self.batch_size, self.z_dim)).astype(np.float32)  
            
                # Update D network
                _ = self.sess.run([d_optim],
                    feed_dict = dict({ 
                        self.inputs: batch_images
                    }, **feed_dict_z))    
               
                # Update G network
                _ = self.sess.run([g_optim],
                    feed_dict = dict({ 
                    }, **feed_dict_z))
                    
                # Update G network
                _ = self.sess.run([g_optim],
                    feed_dict = dict({ 
                    }, **feed_dict_z))

                errD_fake = self.d_loss_fake.eval(dict({}, **feed_dict_z))
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval(dict({}, **feed_dict_z))      

                counter += 1
                print("Epoch: [%2d] [%8d/%8d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake + errD_real, errG))
                
                if np.mod(counter, 3165) == 0:
                    sample_inputs = batch_images
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict=dict({
                            self.inputs: sample_inputs,
                        }, **sample_feed_dict_z))
                    save_images(samples[0 : 64], image_manifold_size(samples[0 : 64].shape[0]),
                        '{}/train_{:02d}_{:08d}.png'.format(self.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                    
                 
                if np.mod(counter, 3165) == 0:
                    self.save(counter)
                        
    def save(self, step):
        model_name = "DCGAN.model"
        self.saver.save(self.sess,
            os.path.join(self.checkpoint_dir, model_name),
            global_step=step)
                        
    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
     
    def get_collisions(self, nbatch, topK, runs):
        could_load, checkpoint_counter = self.load()
        print(could_load, checkpoint_counter)              
        for ii in range(runs):
            samples = []
            z_space = []
            for i in range(nbatch):
                feed_dict_z = {}
                feed_dict_z[self.z[0]] = np.random.normal(size = (self.batch_size, self.z_dim)).astype(np.float32)  
                z_space.append(feed_dict_z[self.z[0]])
                samples.append(self.sess.run(self.sampler,
                    feed_dict=dict({ 
                    }, **feed_dict_z)
                    ))
            images = np.concatenate(samples)
            z_space1 = np.concatenate(np.array(z_space).astype(np.float32))
            queue = []
            n_image = images.shape[0]

            for i in range(n_image):
                for j in range(i+1, n_image):
                    dist = np.sum((images[i] - images[j])**2)
                    dist_z = np.sum((z_space1[i] - z_space1[j])**2)
                    if len(queue) == 0 or -1*dist > queue[0][0]:
                        hq.heappush(queue, (-1*copy.deepcopy(dist), copy.deepcopy(dist_z), copy.deepcopy(images[i]), copy.deepcopy(images[j])))
                        if len(queue) > topK:
                            hq.heappop(queue)
            f_images = []
            for idx in range(topK):
                neg_dist, dist_z, img1, img2 = hq.heappop(queue)
                f_images.append(img1)
                f_images.append(img2)
                print -neg_dist, dist_z
                #scipy.misc.imsave(self.sample_dir + '/pair#%d_%f_%d.png'%(idx, -1*neg_dist, 1), (img1+1.)/2)
                #scipy.misc.imsave(self.sample_dir + '/pair#%d_%f_%d.png'%(idx, -1*neg_dist, 2), (img2+1.)/2)                
            f_images = np.asarray(f_images)
            save_images(f_images[0 : 2*topK], image_manifold_size_pairs(f_images[0 : 2*topK].shape[0]),
                    '{}/top_{}_pairs_from_{:01d}_run_{:01d}.png'.format(self.sample_dir,topK,nbatch*64,ii))
            print "\n"

          
            
            
    