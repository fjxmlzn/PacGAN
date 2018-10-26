# PacGAN: The power of two samples in generative adversarial networks

**[[paper (arXiv)](https://arxiv.org/abs/1712.04086)]**
**[[website](http://swoh.web.engr.illinois.edu/pacgan.html)]**
**[[interview (youtube)](https://www.youtube.com/watch?v=MqdhDdD4-Z0)]**
**[[code](https://github.com/fjxmlzn/PacGAN)]**


**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Ashish Khetan](http://web.engr.illinois.edu/~khetan2/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/), [Sewoong Oh](http://web.engr.illinois.edu/~swoh/)

## CelebA experiments

### Prerequisites
The codes are based on [Taehoon Kim](https://carpedm20.github.io/)'s [implementation](https://github.com/carpedm20/DCGAN-tensorflow) of [DCGAN](https://arxiv.org/abs/1511.06434). Before running codes, please download CelebA dataset according to instructions on [this page](https://github.com/carpedm20/DCGAN-tensorflow), and put *.jpg files under `./celebA/img_align_celeba` folder. Many thanks to Taehoon Kim!

Before running codes, you may need to change GPU configurations according to the devices you have. The configurations are set in `config.py` in each directory. Please refer to [GPUTaskScheduler github page](https://github.com/fjxmlzn/GPUTaskScheduler) for details of how to make proper configurations.

### Running code
```
python main.py
```
