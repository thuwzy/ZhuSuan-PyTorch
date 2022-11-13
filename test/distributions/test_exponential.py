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
from zhusuan.distributions.exponential import Exponential

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestExponential(unittest.TestCase):

    def test_init(self):
        exp = Exponential(1.)
        self.assertEqual(exp._dtype, torch.float32)
        Exponential(torch.tensor([1., 2., 3.]))
        with self.assertRaisesRegex(TypeError, r"must have a dtype in"):
            Exponential(0, dtype=torch.int64)

    def test_property(self):
        rate = torch.rand([2, 2]).abs_()
        exp = Exponential(rate=rate)
        self.assertTrue(exp.rate.equal(rate))
        sample = exp.sample()
        self.assertTrue(torch.norm(torch.log(exp.prob(sample)) - exp.log_prob(sample)) < 1e-6)


    def test_dtype(self):
        utils.test_float_dtype_1parameter_discrete(self, Exponential, allow_16bit=False)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_same(self, Exponential, torch.ones)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(self, Exponential, torch.ones, True)

    def test_log_porb_shape(self):
        utils.test_1parameter_log_prob_shape_same(self, Exponential, torch.ones, torch.ones)

    def test_value(self):
        def _test_value(beta, given):
            beta = np.array(beta, dtype=np.float32)
            log_p = Exponential(beta).log_prob(given)
            # when alpha == 1., gamma distribution is same with Exponential distribution
            target_log_p = stats.gamma.logpdf(given, 1., scale=1 / beta)
            np.testing.assert_allclose(log_p.numpy(), target_log_p, rtol=1e-03)

        _test_value([2.], [1.])
        _test_value([10., 3, 6.7], [2., 4., 6.])

    def test_distribution_shape(self):
        utils.test_and_save_distribution_img(Exponential(1.))