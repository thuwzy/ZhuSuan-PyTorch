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
from zhusuan.distributions.laplace import Laplace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestLaplace(unittest.TestCase):

    def test_init(self):
        lap = Laplace(0.1, 0.2)
        self.assertEqual(lap.loc, torch.tensor(0.1))
        self.assertEqual(lap.scale, torch.tensor(0.2))
        self.assertEqual(lap._dtype, torch.float32)
        lap = Laplace(torch.tensor([1., 2.]), torch.tensor([[1., 2.], [2., 3.]]))
        self.assertTrue(lap.loc.equal(torch.tensor([1., 2.])))

        with self.assertRaisesRegex(TypeError, r"must have a dtype in"):
            Laplace(loc=2, scale=2, dtype=torch.int64)

        # make sure broadcast pre-check
        with self.assertRaises(RuntimeError):
            Laplace(torch.zeros([2, 1]), torch.zeros([2, 4, 3]))

    def test_property(self):
        loc = torch.rand([2, 2]).abs_()
        scale = torch.rand([2, 2]).abs_()
        la = Laplace(loc, scale)
        self.assertTrue(loc.equal(la.loc))
        self.assertTrue(la.scale.equal(scale))
        sample = la.sample()
        self.assertTrue(torch.norm(torch.log(la.prob(sample)) - la.log_prob(sample)) < 1e-6)

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Laplace)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Laplace, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Laplace, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, Laplace, torch.ones, torch.ones, torch.ones)


    def test_value(self):
        def _test_value(loc, scale, given):
            log_p = Laplace(loc, scale).log_prob(given)
            target_log_p = stats.laplace.logpdf(given, loc, scale)
            np.testing.assert_allclose(log_p.numpy(), target_log_p, rtol=1e-03)

        _test_value([2.], [1.], [3.])
        _test_value([3., 4.], [1., 100.], [3., 100.])

        with self.assertRaises(ValueError):
            _test_value([2.], [-1.], [20.])



    def test_distribution_shape(self):
        utils.test_and_save_distribution_img(Laplace(0., 2.))