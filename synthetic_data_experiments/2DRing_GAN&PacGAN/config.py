from blocks.bricks import Rectifier
from blocks.initialization import IsotropicGaussian, Constant
import itertools, numpy
from math import sin, cos

MEANS = [numpy.array([cos(id * numpy.pi * 2 / 8), sin(id * numpy.pi * 2 / 8)]) for id in range(8)]
VARIANCES = [0.01 ** 2 * numpy.eye(len(mean)) for mean in MEANS]

config = {
    "scheduler_config": {
        "backend": "theano",
        "gpu": ["gpu0", "gpu1", "gpu2", "gpu3"]
    },

    "theano_config": {
        "theanorc_template_file": ".theanorc"
    },

    "global_config": {
        "num_epoch": 400,
        "num_xdim": 2,
        "gen_hidden_size": 400,
        "disc_hidden_size": 200,
        "gen_activation": Rectifier,
        "disc_maxout_pieces": 5,
        "weights_init_std": 0.02,
        "biases_init": Constant(0.0),
        "learning_rate": 1e-4,
        "beta1": 0.8,
        "batch_size": 100,
        "monitoring_batch_size": 500,
        "x_mode_means": MEANS,
        "x_mode_variances": VARIANCES,
        "x_mode_priors": None,
        "num_sample": 100000,
        "main_loop_file": "main_loop.tar",
        "num_log_figure_sample": 2500,
        "num_zmode": 1,
        "num_zdim": 2,
        "z_mode_r": 0.0,
        "z_mode_std": 1.0,
        "log_models": False,
        "log_figures": True,
        "log_metrics": True,
        "fuel_random": True,
        "blocks_random": True,
    },

    "test_config": [
        {
            "num_packing": [1, 2, 3, 4],
            "run": range(0, 10)
        }
    ]
}