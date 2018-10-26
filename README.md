# PacGAN: The power of two samples in generative adversarial networks

**[[paper (arXiv)](https://arxiv.org/abs/1712.04086)]**
**[[website](http://swoh.web.engr.illinois.edu/pacgan.html)]**
**[[interview (youtube)](https://www.youtube.com/watch?v=MqdhDdD4-Z0)]**
**[[code](https://github.com/fjxmlzn/PacGAN)]**


**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Ashish Khetan](http://web.engr.illinois.edu/~khetan2/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/), [Sewoong Oh](http://web.engr.illinois.edu/~swoh/)

**Abstract:** Generative adversarial networks (GANs) are innovative techniques for learning generative models of complex data distributions from samples. Despite remarkable recent improvements in generating realistic images, one of their major shortcomings is the fact that in practice, they tend to produce samples with little diversity, even when trained on diverse datasets. This phenomenon, known as mode collapse, has been the main focus of several recent advances in GANs. Yet there is little understanding of why mode collapse happens and why existing approaches are able to mitigate mode collapse. We propose a principled approach to handling mode collapse, which we call packing. The main idea is to modify the discriminator to make decisions based on multiple samples from the same class, either real or artificially generated. We borrow analysis tools from binary hypothesis testing---in particular the seminal result of Blackwell [Bla53]---to prove a fundamental connection between packing and mode collapse. We show that packing naturally penalizes generators with mode collapse, thereby favoring generator distributions with less mode collapse during the training process. Numerical experiments on benchmark datasets suggests that packing provides significant improvements in practice as well.

## Codes for reproducing results in paper

### Prerequisites
The codes are based on [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler) library. Please install it first.

### Code list
* [Synthetic data experiments](https://github.com/fjxmlzn/PacGAN/tree/master/synthetic_data_experiments)
* [Stacked MNIST experiments](https://github.com/fjxmlzn/PacGAN/tree/master/stacked_MNIST_experiments) 
* [CelebA experiments](https://github.com/fjxmlzn/PacGAN/tree/master/celeba_experiments) 
