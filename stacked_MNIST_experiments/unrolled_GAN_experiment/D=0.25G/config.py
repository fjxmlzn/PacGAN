import numpy as np

config = {
    "scheduler_config": {
        "gpu": ["0", "1", "2", "3"]
    },

    "global_config": {
        "epoch": 1,
        "disc_learning_rate": 0.0002,
        "gen_learning_rate": 0.0001,
        "beta1": 0.5,
        "beta2": 0.990,
        "batch_size": 128,
        "input_width": None,
        "output_width": None,
        "visualize": False,
        "random": True,
        "num_test_sample": 25600,
        "log_metrics": True,
        "num_training_sample": 500000 * 128, # The experiments in Unrolled GAN paper generate data on the fly, so we do it this way.
        "z_dim": 256,
        "dataset": "mnist"
    },

    "test_config": [
        {
            "run": range(50),
            "input_height": [28],
            "output_height": [28],
            "packing_num": [1, 2, 3, 4]
        }
    ]
}