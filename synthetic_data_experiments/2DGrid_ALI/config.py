config = {
    "scheduler_config": {
        "backend": "theano",
        "gpu": ["gpu0", "gpu1", "gpu2", "gpu3"]
    },

    "theano_config": {
        "theanorc_template_file": ".theanorc"
    },

    "global_config": {
    },

    "test_config": [
        {
            "run": range(0, 10)
        }
    ]
}