#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import unittest
from scipy import stats
from test.distributions import utils
from zhusuan.distributions.gamma import Gamma

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestGamma(unittest.TestCase):

    def test_init(self):
        gamma = Gamma(0.1, 0.2)
        self.assertEqual(gamma.beta, torch.tensor(0.2))
        self.assertEqual(gamma.alpha, torch.tensor(0.1))
        self.assertEqual(gamma._dtype, torch.float32)
        gamma = Gamma(torch.tensor([1., 2.]), torch.tensor([[1., 2.], [2., 3.]]))
        self.assertTrue(gamma.alpha.equal(torch.tensor([1., 2.])))

        with self.assertRaisesRegex(TypeError, r"must have a dtype in"):
            Gamma(alpha=2, beta=2, dtype=torch.int64)

        # make sure broadcast pre-check
        with self.assertRaises(RuntimeError):
            Gamma(torch.zeros([2, 1]), torch.zeros([2, 4, 3]))

    def test_property(self):
        alpha = torch.rand([2, 2]).abs_()
        beta = torch.rand([2, 2]).abs_()
        be = Gamma(alpha, beta)
        self.assertTrue(alpha.equal(be.alpha))
        self.assertTrue(be.beta.equal(beta))
        sample = be.sample()
        self.assertTrue(torch.norm(torch.log(be._prob(sample)) - be.log_prob(sample)) < 1e-6)


    def test_dtype(self):
        utils.test_dtype_2parameter(self, Gamma)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Gamma, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Gamma, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, Gamma, torch.ones, torch.ones, torch.ones)

    def test_value(self):
        def _test_value(alpha, beta, given):
            beta = np.array(beta, dtype=np.float32)
            log_p = Gamma(alpha, beta).log_prob(given)
            target_log_p = stats.gamma.logpdf(given, alpha, scale=1 / beta)
            np.testing.assert_allclose(log_p.numpy(), target_log_p, rtol=1e-03)

        _test_value([1.], [1.], [0.6])
        _test_value([2., 1.], [3., 1.], [2., 3.])
        with self.assertRaises(ValueError):
            # raise when beta <=0
            _test_value([1.], [0.], [0.6])
            _test_value([1.], [-1.], [0.6])


    def test_distribution_shape(self):
        dis = Gamma(1.5, 0.2)
        utils.test_and_save_distribution_img(dis)