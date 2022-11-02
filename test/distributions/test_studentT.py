#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import unittest
from test.distributions import utils
from zhusuan.distributions.studentT import StudentT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestStudentT(unittest.TestCase):

    def test_init(self):
        stu = StudentT(1., 0.1, 0.2)
        self.assertEqual(stu.df, torch.tensor(1.))
        self.assertEqual(stu.loc, torch.tensor(0.1))
        self.assertEqual(stu.scale, torch.tensor(0.2))
        self.assertEqual(stu._dtype, torch.float32)
        stu = StudentT(torch.tensor([1., 2.]), torch.tensor([[1., 2.], [2., 3.]]))
        self.assertTrue(stu.loc.equal(torch.tensor([[1., 2.], [2., 3.]])))

        with self.assertRaisesRegex(TypeError, r"must have a dtype in"):
            StudentT(2., loc=2, scale=2, dtype=torch.int64)

    def test_dtype(self):
        utils.test_dtype_3parameter(self, StudentT)

    def test_batch_shape(self):
        utils.test_batch_shape_3parameter_univariate(self, StudentT, torch.ones, torch.ones, torch.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(self, StudentT, torch.ones, torch.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(self, StudentT, torch.ones, torch.ones, torch.ones)







    def test_distribution_shape(self):
        dis = StudentT(10., 0., 1.)
        utils.test_and_save_distribution_img(dis)