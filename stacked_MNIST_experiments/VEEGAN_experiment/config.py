import numpy as np

config = {
    "scheduler_config": {
        "gpu": ["0", "1", "2", "3"]
    },

    "global_config": {
        "epoch": 50,
        "learning_rate": 0.0002,
        "beta1": 0.5,
        "train_size": np.inf,
        "batch_size": 64,
        "input_width": None,
        "output_width": None,
        "input_fname_pattern": "*.png",
        "crop": False,
        "visualize": False,
        "random": True,
        "num_test_sample": 26000,
        "log_metrics": True,
        "num_training_sample": 128000,
        "z_dim": 100,
        "dataset": "mnist"
    },

    "test_config": [
        {
            "run": range(10),
            "input_height": [28],
            "output_height": [28],
            "packing_num": [1, 2, 3, 4]
        }
    ]
}