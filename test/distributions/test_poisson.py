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
from zhusuan.distributions.poisson import Poisson

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestPoisson(unittest.TestCase):
    def test_init(self):
        # default is float32
        poisson = Poisson(rate=1.2)
        self.assertEqual(poisson._dtype, torch.float32)

        poisson = Poisson(rate=torch.tensor([1.2, 2.2], dtype=torch.float64))
        self.assertEqual(poisson._dtype, torch.float64)

    def test_dtype(self):
        utils.test_float_dtype_1parameter_discrete(self, Poisson, allow_16bit=False)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_same(self, Poisson, torch.ones)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(self, Poisson, torch.ones, True)

    def test_log_porb_shape(self):
        utils.test_1parameter_log_prob_shape_same(self, Poisson, torch.ones, torch.ones)

    def test_value(self):
        def _test_value(l, given):
            log_p = Poisson(l).log_prob(given)
            target_log_p = stats.poisson.logpmf(given, l)
            np.testing.assert_allclose(log_p.numpy(), target_log_p, rtol=1e-03)

        _test_value([2.], [1.])
        _test_value([2., 8.], [1., 3.])

        with self.assertRaises(ValueError):
            # raise when float given and minus rate
            _test_value([2.], [1.1])
            _test_value([-2.], [1.1])


    def test_property(self):
        rate = torch.tensor([3.3, 2.2, 9909.7])
        poisson = Poisson(rate)
        self.assertTrue(rate.equal(poisson._rate))
        given = torch.ones([3])
        porb = poisson._prob(given)
        self.assertEqual(porb.shape, given.shape)
        poisson.sample()

        log_porb = poisson._log_prob()
        self.assertEqual(log_porb.shape, rate.shape)

    def test_distribution_shape(self):
        utils.test_and_save_distribution_img(Poisson(0.3))