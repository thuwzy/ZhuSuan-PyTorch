#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import unittest
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

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Gamma)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Gamma, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Gamma, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, Gamma, torch.ones, torch.ones, torch.ones)
