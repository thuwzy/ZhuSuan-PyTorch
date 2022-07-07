#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import unittest
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

    def test_dtype(self):
        utils.test_float_dtype_1parameter_discrete(self, Exponential, allow_16bit=False)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_same(self, Exponential, torch.ones)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(self, Exponential, torch.ones, True)

    def test_log_porb_shape(self):
        utils.test_1parameter_log_prob_shape_same(self, Exponential, torch.ones, torch.ones)

    def test_property(self):
        exp = Exponential(rate=0.1)
        self.assertEqual(exp.rate, torch.tensor(0.1))
