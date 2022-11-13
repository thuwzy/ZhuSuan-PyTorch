#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from scipy import stats
import unittest
from test.distributions import utils
from zhusuan.distributions.logistic import Logistic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestLogistic(unittest.TestCase):

    def test_init(self):
        logistic = Logistic(0.1, 0.2)
        self.assertEqual(logistic.loc, torch.tensor(0.1))
        self.assertEqual(logistic.scale, torch.tensor(0.2))
        self.assertEqual(logistic._dtype, torch.float32)
        logistic = Logistic(torch.tensor([1., 2.]), torch.tensor([[1., 2.], [2., 3.]]))
        self.assertTrue(logistic.loc.equal(torch.tensor([1., 2.])))

        with self.assertRaisesRegex(TypeError, r"must have a dtype in"):
            Logistic(loc=2, scale=2, dtype=torch.int64)
        # make sure broadcast pre-check
        with self.assertRaises(RuntimeError):
            Logistic(torch.ones([2, 1]), torch.ones([2, 4, 3]))

    def test_property(self):
        loc = torch.rand([2, 2]).abs_()
        scale = torch.rand([2, 2]).abs_()
        la = Logistic(loc, scale)
        self.assertTrue(loc.equal(la.loc))
        self.assertTrue(la.scale.equal(scale))
        sample = la.sample()
        self.assertTrue(torch.norm(torch.log(la._prob(sample)) - la.log_prob(sample)) < 1e-6)

    def test_sample_reparameterized(self):
        loc = torch.rand([2, 3], requires_grad=True)
        scale = torch.rand([2, 3]).abs_().requires_grad_()

        logistic = Logistic(loc, scale)
        sample = logistic.sample()
        loc_grad, scale_grad = torch.autograd.grad(
            outputs=sample.sum(), inputs=[loc, scale],
            allow_unused=True
        )
        self.assertTrue(loc_grad is not None)
        self.assertTrue(scale_grad is not None)


    def test_dtype(self):
        utils.test_dtype_2parameter(self, Logistic)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Logistic, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Logistic, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, Logistic, torch.ones, torch.ones, torch.ones)

    def test_value(self):
        def _test_value(loc, scale, given):
            log_p = Logistic(loc, scale).log_prob(given)
            target_log_p = stats.logistic.logpdf(given, loc, scale)
            np.testing.assert_allclose(log_p.numpy(), target_log_p, rtol=1e-03)

        _test_value([2.], [1.], [3.])
        with self.assertRaises(ValueError):
            # raise when scale less equal than 0
            _test_value([2.], [-1.], [3.])


    def test_distribution_shape(self):
        utils.test_and_save_distribution_img(Logistic(0., 1.))
