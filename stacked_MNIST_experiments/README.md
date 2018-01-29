# PacGAN: The power of two samples in generative adversarial networks

[[paper (arXiv)](https://arxiv.org/abs/1712.04086)] [[code](https://github.com/fjxmlzn/PacGAN)]

**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Ashish Khetan](http://web.engr.illinois.edu/~khetan2/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/), [Sewoong Oh](http://web.engr.illinois.edu/~swoh/)

## Stacked MNIST experiments

### Prerequisites
The codes are based on [Taehoon Kim](https://carpedm20.github.io/)'s [implementation](https://github.com/carpedm20/DCGAN-tensorflow) of [DCGAN](https://arxiv.org/abs/1511.06434). Before running codes, please download MNIST dataset according to instructions on [this page](https://github.com/carpedm20/DCGAN-tensorflow). Many thanks to Taehoon Kim!

Before running codes, you may need to change GPU configurations according to the devices you have. The configurations are set in `config.py` in each directory. Please refer to [GPUTaskScheduler github page](https://github.com/fjxmlzn/GPUTaskScheduler) for details of how to make proper configurations.

### Running code
* [VEEGAN](https://arxiv.org/abs/1705.07761) experiment
```
cd VEEGAN_experiment
python train_mnist.py
python main.py
```

* [Unrolled GAN](https://arxiv.org/abs/1611.02163) experiment, D=1/2G
```
cd unrolled_GAN_experiment
cd D=0.5G
python train_mnist.py
python main.py
```

* [Unrolled GAN](https://arxiv.org/abs/1611.02163) experiment, D=1/4G
```
cd unrolled_GAN_experiment
cd D=0.25G
python train_mnist.py
python main.py
```

> **Note**: The MNIST classifier code `train_mnist.py` is based on Keras' [demo code](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py).

### Results
![Figure3](https://raw.githubusercontent.com/fjxmlzn/PacGAN/master/stacked_MNIST_experiments/results/Figure3.png)

![Table2](https://raw.githubusercontent.com/fjxmlzn/PacGAN/master/stacked_MNIST_experiments/results/Table2.png)

![Table3](https://raw.githubusercontent.com/fjxmlzn/PacGAN/master/stacked_MNIST_experiments/results/Table3.png)

The detailed explanation of the architectures, hyperparameters, metrics, and experimental settings are given in the paper.