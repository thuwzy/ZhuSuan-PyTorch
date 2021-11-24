# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from scipy import stats
import numpy as np
import six

from zhusuan import mcmc
from zhusuan.framework import BayesianNet

import unittest


class TestNode:
    def __init__(self, x):
        self.tensor = x


class Test_Model(BayesianNet):
    def __init__(self, x):
        super().__init__()
        self.nodes['x'] = TestNode(x)

    def forward(self, observed):
        self.observe(observed)
        return self

    def log_joint(self, use_cache=False):
        x = self.observed['x']
        x.requires_grad = True # !check
        lh_noise = torch.normal(mean=0., std=2., size= x.shape)
        res = 2 * torch.pow(x, 2) - torch.pow(x, 4) + lh_noise
        return res.sum()


def sample_error_with(sampler, n_chains=1, n_iters=80000, thinning=50, burinin=None, dtype=torch.float32,
                      sampler_type='hmc'):
    if burinin is None:
        burinin = n_iters * 2 // 3
    x = torch.zeros([n_chains], dtype=dtype)
    model = Test_Model(x)
    samples = []
    for t in range(n_iters):
        if sampler_type == 'sgld':
            resample = True if t == 0 else False
            x_sample = sampler.sample(model, {}, resample)['x'].detach().numpy()
        else:
            x_sample = sampler.sample(model, {}, {'x': x})['x'].detach().numpy()
        if np.isnan(x_sample.sum()):
            raise ValueError("nan encountered")
        if t >= burinin and t % thinning == 0:
            samples.append(x_sample)
    samples = np.array(samples)
    samples = samples.reshape(-1)
    A = 3
    xs = np.linspace(-A, A, 1000)
    pdfs = np.exp(2 * (xs ** 2) - xs ** 4)
    pdfs = pdfs / pdfs.mean() / A / 2
    est_pdfs = stats.gaussian_kde(samples)(xs)
    return np.abs(est_pdfs - pdfs).mean()


# class TestMCMC(unittest.TestCase):
#     def test_hmc(self):
#         sampler = mcmc.HMC(step_size=0.01, n_leapfrogs=10)
#         e = sample_error_with(sampler, n_chains=100, n_iters=1000)
#         print(e)


class TestSGMCMC(unittest.TestCase):
    def test_sgld(self):
        sampler = mcmc.SGLD(learning_rate=0.01)
        e = sample_error_with(sampler, n_chains=100, n_iters=8000, sampler_type='sgld')
        print("the result is :", e)
        assert(e==0)

    # def test_psgld(self):
    #     sampler = mcmc.PSGLD(learning_rate=0.01)
    #     e = sample_error_with(sampler, n_chains=100, n_iters=8000, sampler_type='sgld')
    #     print(e)

    # def test_sghmc(self):
    #     sampler = mcmc.SGHMC(learning_rate=0.01, n_iter_resample_v=50,
    #                          friction=0.3, variance_estimate=0.02,
    #                          second_order=False)
    #     e = sample_error_with(sampler, n_chains=100, n_iters=8000, sampler_type='sgld')
    #     print(e)

    # def test_sghmc_second_order(self):
    #     sampler = mcmc.SGHMC(learning_rate=0.01, n_iter_resample_v=50,
    #                          friction=0.3, variance_estimate=0.02,
    #                          second_order=True)
    #     e = sample_error_with(sampler, n_chains=100, n_iters=8000, sampler_type='sgld')
    #     print(e)
