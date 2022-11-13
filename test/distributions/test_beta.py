#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import unittest
import numpy as np
from scipy import stats
from test.distributions import utils
from zhusuan.distributions.beta import Beta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestBeta(unittest.TestCase):

    def test_init(self):
        beta = Beta(0.1, 0.2)
        self.assertEqual(beta.beta, torch.tensor(0.2))
        self.assertEqual(beta.alpha, torch.tensor(0.1))
        beta = Beta(torch.tensor([1., 2.]), torch.tensor([[1., 2.], [2., 3.]]))
        self.assertTrue(beta.alpha.equal(torch.tensor([1., 2.])))

        # make sure broadcast pre-check
        with self.assertRaises(RuntimeError):
            Beta(torch.zeros([2, 1]), torch.zeros([2, 4, 3]))

    def test_property(self):
        alpha = torch.rand([2, 2]).abs_()
        beta = torch.rand([2, 2]).abs_()
        be = Beta(alpha, beta)
        self.assertTrue(alpha.equal(be.alpha))
        self.assertTrue(be.beta.equal(beta))
        sample = be.sample()
        self.assertTrue(torch.norm(torch.log(be._prob(sample)) - be.log_prob(sample)) < 1e-6)

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Beta)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Beta, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Beta, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, Beta, torch.ones, torch.ones, torch.ones)

    def test_value(self):
        def _test_value(alpha, beta, given):
            log_p = Beta(alpha, beta).log_prob(given)
            target_log_p = stats.beta.logpdf(given, alpha, beta)
            np.testing.assert_allclose(log_p.numpy(), target_log_p, rtol=1e-03)

        _test_value([0.5], [0.5], [0.2])
        with self.assertRaises(ValueError):
            Beta([0.5], [0.5]).log_prob([2.])

        # TODO: more value examples

    def test_distribution_shape(self):
        utils.test_and_save_distribution_img(Beta(2., 2.))
