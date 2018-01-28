from theano.tensor import join
from math import sin, cos, pi
from theano.sandbox.rng_mrg import MRG_RandomStreams
from fuel import config

default_rng = None

def init_rng():
    global default_rng
    default_rng = MRG_RandomStreams(seed = config.default_seed)

def circle_gaussian_mixture(num_modes, num_samples, dimension, r=0.0, std=1.0, theano_rng=None):
    global default_rng
    if theano_rng is None:
        if default_rng is None:
            init_rng()
        theano_rng = default_rng
    samples = None
    if dimension == 1:
        for i in range(num_modes):
            num_samples_local = (num_samples + i) // num_modes
            avg = -r + 2 * r / max(1, num_modes - 1) * i
            samples_local = theano_rng.normal((num_samples_local, dimension), avg=avg, std=std)
            if samples is None:
                samples = samples_local
            else:
                samples = join(0, samples, samples_local)
    elif dimension >= 2:
        for i in range(num_modes):
            num_samples_local = (num_samples + i) // num_modes
            x = r * cos(2 * pi / num_modes * i)
            y = r * sin(2 * pi / num_modes * i)
            samples_local_x = theano_rng.normal((num_samples_local, 1), avg=0.0, std=std)
            samples_local_x += x
            samples_local_y = theano_rng.normal((num_samples_local, 1), avg=0.0, std=std)
            samples_local_y += y
            samples_local = join(1, samples_local_x, samples_local_y)
            if dimension > 2:
                samples_local_left = theano_rng.normal((num_samples_local, dimension - 2), avg=0.0, std=std)
                samples_local = join(1, samples_local, samples_local_left)
            
            if samples is None:
                samples = samples_local
            else:
                samples = join(0, samples, samples_local)
    return samples
    
if __name__ == "__main__":
    from matplotlib import pyplot
    
    samples1 = circle_gaussian_mixture(num_modes=1, num_samples=10000, dimension=1).eval()
    pyplot.figure(1)
    pyplot.hist(samples1.ravel(), bins=50)
    pyplot.show()
    
    samples2 = circle_gaussian_mixture(num_modes=3, num_samples=10000, dimension=1, r=2, std=0.1).eval()
    pyplot.figure(2)
    pyplot.hist(samples2.ravel(), bins=50)
    pyplot.show()
    
    samples3 = circle_gaussian_mixture(num_modes=3, num_samples=2500, dimension=2, r=3, std=0.2).eval()
    fig = pyplot.figure(3)
    pyplot.scatter(samples3[:, 0], samples3[:, 1], marker='.', c='black', alpha=0.3)
    pyplot.xlim(-4, 4)
    pyplot.ylim(-4, 4)
    pyplot.show()
    fig.savefig('3.png')
    
    samples4 = circle_gaussian_mixture(num_modes=25, num_samples=2500, dimension=2, r=3, std=0.2).eval()
    fig = pyplot.figure(4)
    pyplot.scatter(samples4[:, 0], samples4[:, 1], marker='.', c='black', alpha=0.3)
    pyplot.xlim(-4, 4)
    pyplot.ylim(-4, 4)
    pyplot.show()
    fig.savefig('25.png')
    
    samples5 = circle_gaussian_mixture(num_modes=1, num_samples=2500, dimension=2, r=0, std=0.2).eval()
    fig = pyplot.figure(5)
    pyplot.scatter(samples5[:, 0], samples5[:, 1], marker='.', c='black', alpha=0.3)
    pyplot.xlim(-4, 4)
    pyplot.ylim(-4, 4)
    pyplot.show()
    fig.savefig('1.png')
    
    samples6 = circle_gaussian_mixture(num_modes=2, num_samples=2500, dimension=2, r=3.0, std=0.2).eval()
    fig = pyplot.figure(6)
    pyplot.scatter(samples6[:, 0], samples6[:, 1], marker='.', c='black', alpha=0.3)
    pyplot.xlim(-4, 4)
    pyplot.ylim(-4, 4)
    pyplot.show()
    fig.savefig('2.png')
        