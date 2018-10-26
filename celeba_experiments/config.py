import numpy as np

config = {
    "scheduler_config": {
        "gpu": ["0"],
        "force_rerun": False 
    },

    "global_config": {
    },

    "test_config": [
        {
            "run": range(2),
            "packing_num": [1,2,3,4]
        }
    ]
}