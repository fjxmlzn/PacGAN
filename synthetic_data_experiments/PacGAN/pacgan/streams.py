"""Functions for creating data streams."""
from fuel.datasets import CIFAR10, SVHN, CelebA
from fuel.datasets.toy import Spiral
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

from datasets import GaussianPackingMixture, VEEGAN1200DPackingMixture

def create_packing_VEEGAN1200D_data_streams(num_packings, batch_size, monitoring_batch_size, rng=None, num_examples=100000, sources=('features', )):

    train_set = VEEGAN1200DPackingMixture(num_packings=num_packings, num_examples=num_examples, rng=rng, sources=sources)

    valid_set = VEEGAN1200DPackingMixture(num_packings=num_packings, num_examples=num_examples, rng=rng, sources=sources)

    main_loop_stream = DataStream(train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size=batch_size, rng=rng))

    train_monitor_stream = DataStream(train_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    valid_monitor_stream = DataStream(valid_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    return main_loop_stream, train_monitor_stream, valid_monitor_stream

    

def create_packing_gaussian_mixture_data_streams(num_packings, batch_size, monitoring_batch_size, means=None, variances=None, priors=None, rng=None, num_examples=100000, sources=('features', )):

    train_set = GaussianPackingMixture(num_packings=num_packings, num_examples=num_examples, means=means, variances=variances, priors=priors, rng=rng, sources=sources)

    valid_set = GaussianPackingMixture(num_packings=num_packings, num_examples=num_examples, means=means, variances=variances, priors=priors, rng=rng, sources=sources)

    main_loop_stream = DataStream(train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size=batch_size, rng=rng))

    train_monitor_stream = DataStream(train_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    valid_monitor_stream = DataStream(valid_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    return main_loop_stream, train_monitor_stream, valid_monitor_stream
