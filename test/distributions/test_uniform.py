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
from zhusuan.distributions.uniform import Uniform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestUniform(unittest.TestCase):

    def test_init(self):
        uniform = Uniform(0.1, 0.2)
        self.assertEqual(uniform.low, torch.tensor(0.1))
        self.assertEqual(uniform.high, torch.tensor(0.2))
        self.assertEqual(uniform._dtype, torch.float32)
        uniform = Uniform(torch.tensor([1., 2.]), torch.tensor([[1., 2.], [2., 3.]]))
        self.assertTrue(uniform.low.equal(torch.tensor([1., 2.])))

        with self.assertRaisesRegex(TypeError, r"must have a dtype in"):
            Uniform(2, 2, dtype=torch.int64)

        # make sure broadcast pre-check
        with self.assertRaises(RuntimeError):
            Uniform(torch.zeros([2, 1]), torch.zeros([2, 4, 3]))

    def test_property(self):
        low = torch.rand([2, 2]).abs_()
        high = low + torch.rand([2, 2]).abs_()
        uni = Uniform(low, high)
        self.assertTrue(low.equal(uni.low))
        self.assertTrue(uni.high.equal(high))
        sample = uni.sample()
        self.assertTrue(torch.norm(torch.log(uni._prob(sample)) - uni.log_prob(sample)) < 1e-6)

    def test_sample_reparameterized(self):
        low = torch.rand([2, 3], requires_grad=True)
        high = low.detach() + torch.rand([2, 3]).abs_().requires_grad_()

        uni = Uniform(low, high)
        sample = uni.sample()
        loc_grad, scale_grad = torch.autograd.grad(
            outputs=sample.sum(), inputs=[low, high],
            allow_unused=True
        )
        self.assertTrue(loc_grad is not None)
        self.assertTrue(scale_grad is not None)

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Uniform)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Uniform, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Uniform, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, Uniform, torch.zeros, torch.ones, torch.ones)

    def test_value(self):
        def _test_value(low, high, given):
            low = np.array(low)
            high = np.array(high)
            log_p = Uniform(low, high).log_prob(given)
            target_log_p = stats.uniform.logpdf(given, low, high-low)
            np.testing.assert_allclose(log_p.numpy(), target_log_p, rtol=1e-03)

        _test_value([4.], [5.], [4.5])
        with self.assertRaises(ValueError):
            _test_value([10.], [2.], [3.])

    def test_distribution_shape(self):
        utils.test_and_save_distribution_img(Uniform(-0.5, 0.5))
