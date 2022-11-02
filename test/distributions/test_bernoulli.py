#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import unittest

from test.distributions import utils
from zhusuan.distributions.bernoulli import Bernoulli

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestBernoulli(unittest.TestCase):

    def test_init(self):
        ber = Bernoulli(0.)
        self.assertEqual(ber._dtype, torch.float32)
        self.assertEqual(ber.probs, torch.tensor(0.5))

        ber = Bernoulli(probs=[0.4, 0.5])
        self.assertTrue(ber.logits.equal(torch.log
                                         (ber.probs / (torch.ones(ber.probs.shape) - ber.probs))
                                         ))
        with self.assertRaisesRegex(ValueError, r"Either.*should be passed"):
            Bernoulli(logits=1, probs=0.1)

        with self.assertRaisesRegex(TypeError, r"must have a dtype in"):
            Bernoulli(probs=0, dtype=torch.int64)

    def test_property(self):
        input = [[1, 2], [1, 3]]
        ber = Bernoulli(logits=torch.tensor(input, dtype=torch.float32))
        self.assertEqual(list(ber.logits.shape), [2, 2])
        self.assertEqual(list(ber.probs.shape), [2, 2])

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(self, Bernoulli, torch.ones, True)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_same(self, Bernoulli, torch.ones)

    def test_dtype(self):
        utils.test_float_dtype_1parameter_discrete(self, Bernoulli, allow_16bit=False)

    def test_log_prob_shape(self):
        utils.test_1parameter_log_prob_shape_same(self, Bernoulli, torch.ones, torch.ones)

    def test_value(self):
        pass
        # TODO: get value checked



    def test_distribution_shape(self):
        dis = Bernoulli(probs=0.5)
        utils.test_and_save_distribution_img(dis)