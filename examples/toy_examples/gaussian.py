#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import time
from scipy import stats
import matplotlib.pyplot as plt
import zhusuan as zs
import random
from zhusuan.framework import BayesianNet
from zhusuan.distributions import Normal
from zhusuan.mcmc import SGLD


class Gaussian(BayesianNet):
    def __init__(self, n_x, std, n_particles):
        super(Gaussian, self).__init__()
        self._n_x = n_x
        self._std = std
        self._n_particles = n_particles
        self.dist = Normal(torch.zeros([n_x], dtype=torch.float32), std=self._std)

    def forward(self, observed):
        self.observe(observed)
        self.sn(self.dist, "x", n_samples=self._n_particles)
        return self


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(1)

    # Define model parameters
    n_x = 1
    stdev = 1 / (np.arange(n_x, dtype=np.float32) + 1)
    # HMC parameters
    kernel_width = 0.1
    n_chains = 1
    n_iters = 200
    burnin = n_iters // 2
    n_leapfrogs = 20

    # Build the computation graph
    model = Gaussian(n_x, stdev, n_chains)
    sampler = SGLD(learning_rate=1e-3)
    samples = []
    time_st = time.time()
    print('Sampling...')
    for i in range(n_iters):
        if i % 2 == 0:
            d_time = time.time() - time_st
            time_st = time.time()
            print('step: {}, time: {:4f}s'.format(i, d_time))
        sample_ = sampler.sample(model, {},
                                True if i == 0 else False,
                                step=2000)
        samples.append(sample_['x'].cpu().detach().numpy())

    print('Finished.')

    samples = np.vstack(samples)
    # Check & plot the results
    print('Expected mean = {}'.format(np.zeros(n_x)))
    print('Sample mean = {}'.format(np.mean(samples)))
    print('Expected stdev = {}'.format(stdev))
    print('Sample stdev = {}'.format(np.std(samples)))
    print('Relative error of stdev = {}'.format(
        (np.std(samples) - stdev) / stdev))