#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
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

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Logistic)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Logistic, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Logistic, torch.ones, torch.ones)

    def test_log_prob_shaple(self):
        utils.test_2parameter_log_prob_shape_same(self, Logistic, torch.ones, torch.ones, torch.ones)
