import os, random, pickle, csv
import scipy.misc
import numpy as np

from math import ceil
from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables, save_images, image_manifold_size

import tensorflow as tf

from gpu_task_scheduler.gpu_task import GPUTask

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class PacGANTask(GPUTask):
    #def required_env(self):
    #    return {"THEANO_FLAGS": "device=gpu3"}
        
    def main(self):
        FLAGS = Struct(**self._config)
        if FLAGS.input_width is None:
            FLAGS.input_width = FLAGS.input_height
        if FLAGS.output_width is None:
            FLAGS.output_width = FLAGS.output_height
        
        FLAGS.checkpoint_dir = os.path.join(self._work_dir, "checkpoint")
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        FLAGS.sample_dir = os.path.join(self._work_dir, "samples")
        if not os.path.exists(FLAGS.sample_dir):
            os.makedirs(FLAGS.sample_dir)
            
        FLAGS.work_dir = self._work_dir

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth=True
        
        if FLAGS.random:
            seed = random.randint(1, 100000)        
            np.random.seed(seed)
            with open(os.path.join(self._work_dir, "seed.txt"), "w") as f:
                f.write("{}".format(seed))
                
        t_num_test_samples = int(ceil(float(FLAGS.num_test_sample) / float(FLAGS.batch_size))) * FLAGS.batch_size
                
        test_samples = np.random.uniform(-1, 1, size = (t_num_test_samples, FLAGS.z_dim))

        with tf.Session(config=run_config) as sess:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                packing_num=FLAGS.packing_num,
                num_training_sample=FLAGS.num_training_sample,
                num_test_sample=FLAGS.num_test_sample,
                z_dim=FLAGS.z_dim,
                test_samples=test_samples)

            show_all_variables()

            dcgan.train(FLAGS)
       
            #OPTION = 0
            #visualize(sess, dcgan, FLAGS, OPTION)