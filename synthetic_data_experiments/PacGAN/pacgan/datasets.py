"""Additional dataset classes."""
from __future__ import (division, print_function, )
from collections import OrderedDict
from scipy.stats import multivariate_normal

import numpy as np
import numpy.random as npr

from fuel import config
from fuel.datasets import H5PYDataset, IndexableDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path

from ali.utils import as_array
from ali.datasets import GaussianMixtureDistribution

class VEEGAN1200DPackingMixture(IndexableDataset):
    def __init__(self, num_examples, num_packings, **kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            seed = kwargs.pop('seed', config.default_seed)
            rng = np.random.RandomState(seed)
            
        all_features = []
        all_labels = []
        all_densities = []
        
        MEANS = [-1, -0.5, 0, 0.5, -2, -2.5, 10, 0.25, -13, -5.5]
        
        features = np.zeros((num_examples, 1200))
        features[:, 0 : 700] = rng.normal(loc=0.0, scale=1.0, size=(num_examples, 700))
        labels = rng.randint(0, len(MEANS), size=num_examples)
        features = features + np.asarray([[MEANS[int(label)]] * 1200 for label in labels])
        features = features.astype(np.float32)
        
        for i in range(num_packings):
            ids = np.random.randint(0, num_examples, size=(num_examples))
            all_features.append(features[ids])
            all_labels.append(labels[ids])

        data = OrderedDict([
            ('features', np.hstack(all_features)),
            ('label', all_labels)
        ])

        super(VEEGAN1200DPackingMixture, self).__init__(data, **kwargs) 


class GaussianPackingMixture(IndexableDataset):
    """ Toy dataset containing packing points sampled from a gaussian mixture distribution.

    The dataset contains 3 sources:
    * features
    * label
    * densities

    """
    def __init__(self, num_examples, num_packings, means=None, variances=None, priors=None, **kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            seed = kwargs.pop('seed', config.default_seed)
            rng = np.random.RandomState(seed)
            
        all_features = []
        all_labels = []
        all_densities = []
        
        gaussian_mixture = GaussianMixtureDistribution(means=means, variances=variances, priors=priors, rng=rng)

        features, labels = gaussian_mixture.sample(nsamples=num_examples)
        densities = gaussian_mixture.pdf(x=features)
        
        for i in range(num_packings):
            ids = np.random.randint(0, num_examples, size=(num_examples))
            all_features.append(features[ids])
            all_labels.append(labels[ids])
            all_densities.append(densities[ids])

        data = OrderedDict([
            ('features', np.hstack(all_features)),
            ('label', all_labels),
            ('density', all_densities)
        ])

        super(GaussianPackingMixture, self).__init__(data, **kwargs)
