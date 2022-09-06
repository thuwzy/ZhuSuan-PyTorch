#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import IPython
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
import zhusuan as zs
import random
from zhusuan.framework import BayesianNet
from zhusuan.distributions import Normal
from zhusuan.mcmc import HMC


class Gaussian(BayesianNet):
    def __init__(self, n_x, std, n_particles):
        super(Gaussian, self).__init__()
        self._n_x = n_x
        self._std = std
        self._n_particles = n_particles

    def forward(self, observed):
        self.observe(observed)
        dist = Normal(torch.zeros([n_x], dtype=torch.float32), std=self._std)
        self.sn(dist, "x", n_samples=self._n_particles)
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
    n_chains = 10
    n_iters = 200
    burnin = n_iters // 2
    n_leapfrogs = 5

    # Build the computation graph
    model = Gaussian(n_x, stdev, n_chains)

    samples = []
    print('Sampling...')

    for i in range(n_iters):
        init_x = torch.zeros([n_chains, n_x], requires_grad=False)
        adapt_step_size: bool = i < burnin // 2
        adapt_mass: bool = i < burnin // 2
        hmc = HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs,
                  adapt_step_size=adapt_step_size, adapt_mass=adapt_mass,
                  target_acceptance_rate=0.9)
        sample, hmc_info = hmc.sample(model, {}, {"x": init_x})
        x_sample = hmc_info.samples["x"]
        acc = hmc_info.acceptance_rate
        ss = hmc_info.updated_step_size
        print('Sample {}: Acceptance rate = {}, updated step size = {}'
              .format(i, torch.mean(acc), ss))
        if i >= burnin:
            samples.append(x_sample)
    print('Finished.')

    samples = [sample.detach().numpy() for sample in samples]
    # Check & plot the results
    print('Expected mean = {}'.format(np.zeros(n_x)))
    print('Sample mean = {}'.format(np.mean(samples)))
    print('Expected stdev = {}'.format(stdev))
    print('Sample stdev = {}'.format(np.std(samples)))
    print('Relative error of stdev = {}'.format(
        (np.std(samples) - stdev) / stdev))