#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import unittest
from test.distributions import utils
from zhusuan.distributions.beta import Beta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestBeta(unittest.TestCase):

    def test_init(self):
        beta = Beta(0.1, 0.2)
        self.assertEqual(beta.beta, torch.tensor(0.2))
        self.assertEqual(beta.alpha, torch.tensor(0.1))
        beta = Beta(torch.tensor([1., 2.]), torch.tensor([[1., 2.], [2., 3.]]))
        self.assertTrue(beta.alpha.equal(torch.tensor([1.,2.])))

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Beta)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(self, Beta, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, Beta, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, Beta, torch.ones, torch.ones, torch.ones)






    def test_distribution_shape(self):
        utils.test_and_save_distribution_img(Beta(2.,2.))
