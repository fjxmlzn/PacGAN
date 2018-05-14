# PacGAN: The power of two samples in generative adversarial networks

**[[paper (arXiv)](https://arxiv.org/abs/1712.04086)]**
**[[website](http://swoh.web.engr.illinois.edu/pacgan.html)]**
**[[interview (youtube)](https://www.youtube.com/watch?v=MqdhDdD4-Z0)]**
**[[code](https://github.com/fjxmlzn/PacGAN)]**


**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Ashish Khetan](http://web.engr.illinois.edu/~khetan2/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/), [Sewoong Oh](http://web.engr.illinois.edu/~swoh/)

## Synthetic data experiments

### Prerequisites
The codes are based on [code](https://github.com/IshmaelBelghazi/ALI) of [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704) paper. Please install it first (follow instructions in Requirements and Setup section [here](https://github.com/IshmaelBelghazi/ALI)). Many thanks to ALI's authors!

Before running codes, you may need to change GPU configurations according to the devices you have. The configurations are set in `config.py` in each directory. Please refer to [GPUTaskScheduler's github page](https://github.com/fjxmlzn/GPUTaskScheduler) for details of how to make proper configurations.

### Setup
Install PacGAN library in this folder
```
pip install -e PacGAN
```

### Running code
* GAN & PacGAN on 2DRing dataset
```
cd 2DRing_GAN&PacGAN
python main.py
```

* GAN & PacGAN on 2DGrid dataset
```
cd 2DGrid_GAN&PacGAN
python main.py
```

* [ALI](https://arxiv.org/abs/1606.00704) on 2DRing dataset
```
cd 2DRing_ALI
python main.py
```

* [ALI](https://arxiv.org/abs/1606.00704) on 2DGrid dataset
```
cd 2DGrid_ALI
python main.py
```

### Results
![Figure2](https://github.com/fjxmlzn/PacGAN/blob/master/synthetic_data_experiments/results/Figure2.png)

![Table1](https://github.com/fjxmlzn/PacGAN/blob/master/synthetic_data_experiments/results/Table1.png)

The detailed explanation of the architectures, hyperparameters, metrics, and experimental settings are given in the paper.
