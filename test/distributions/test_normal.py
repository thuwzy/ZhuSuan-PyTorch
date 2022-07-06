#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from scipy import stats
from scipy.special import logsumexp

import unittest

from test.distributions import utils
from zhusuan.distributions.normal import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO: test sample value
class TestNormal(unittest.TestCase):
    def setUp(self):
        self._Normal_std = lambda mean, std, **kwargs: Normal(
            mean=mean, std=std, **kwargs)
        self._Normal_logstd = lambda mean, logstd, **kwargs: Normal(
            mean=mean, logstd=logstd, **kwargs)

    def test_init(self):
        # test given both std and logstd error:
        with self.assertRaisesRegex(ValueError, r"Either.*should be passed"):
            Normal(mean=torch.zeros([2, 1]), std=torch.ones([2, 4, 3]), logstd=torch.zeros([2, 2, 3]))
        # try:
        #     Normal(mean=torch.zeros([2, 1]),
        #            std=torch.ones([2, 4, 3]), logstd=torch.zeros([2, 2, 3]))
        # except:
        #     raise ValueError("Either.*should be passed")

        # try:
        #     Normal(mean=torch.zeros([2, 1]), logstd=torch.zeros([2, 4, 3]))
        # except:
        #     raise ValueError("should be broadcastable to match")

        # try:
        #     Normal(mean=torch.ones([2, 1]), std=torch.ones([2, 4, 3]))
        # except:
        #     raise ValueError("should be broadcastable to match")

        Normal(mean=torch.ones([32, 1], dtype=torch.float32),
               logstd=torch.ones([32, 1, 3], dtype=torch.float32))

        dis = Normal(mean=torch.ones([32, 1], dtype=torch.float32),
                     std=torch.ones([32, 1, 3], dtype=torch.float32))
        self.assertEqual(dis._dtype, torch.float32)

        dis = Normal(mean=torch.ones([2, 1], dtype=torch.float16),
                     std=torch.ones([2, 1], dtype=torch.float16))
        self.assertEqual(dis._dtype, torch.float16)

        std = Normal(mean=0., std=1.)
        self.assertEqual(std._dtype, torch.float32)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(
            self, self._Normal_std, np.zeros, np.ones)
        utils.test_2parameter_sample_shape_same(
            self, self._Normal_logstd, np.zeros, np.zeros)

    def test_batch_shape(self):
        dis = Normal(mean=torch.ones([32, 1], dtype=torch.float32),
                     std=torch.ones([32, 1, 3], dtype=torch.float32))
        dis.batch_shape

    def test_property(self):
        mean = torch.tensor([1., 2.])
        std = torch.tensor([1., 4.])
        dis = Normal(mean=mean, std=std)
        self.assertTrue(mean.equal(dis.mean))
        self.assertTrue(std.equal(dis.std))
        self.assertTrue(torch.log(std).equal(dis.logstd))
        sample = dis.sample()
        self.assertTrue(torch.log(dis._prob(sample)).equal(dis.log_prob(sample)))

    def test_sample_reparameterized(self):
        mean = torch.ones([2, 3])
        logstd = torch.ones([2, 3])
        mean.requires_grad = True
        logstd.requires_grad = True
        norm_rep = Normal(mean=mean, logstd=logstd)
        samples = norm_rep.sample()
        mean_grads, logstd_grads = torch.autograd.grad(outputs=samples.sum(), inputs=[mean, logstd],
                                                       allow_unused=True)
        self.assertTrue(mean_grads is not None)
        self.assertTrue(logstd_grads is not None)

        norm_no_rep = Normal(mean=mean, logstd=logstd, is_reparameterized=False)
        samples = norm_no_rep.sample()
        mean_grads, logstd_grads = torch.autograd.grad(outputs=samples.sum(),
                                                       inputs=[mean, logstd],
                                                       allow_unused=True)

        self.assertEqual(mean_grads.sum(), torch.zeros([1]))
        self.assertEqual(logstd_grads.sum(), torch.zeros([1]))

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(
            self, self._Normal_std, np.zeros, np.ones, np.zeros)
        utils.test_2parameter_log_prob_shape_same(
            self, self._Normal_logstd, np.zeros, np.zeros, np.zeros)

    def test_value(self):
        def _test_value(given, mean, logstd):
            mean = np.array(mean, np.float32)
            given = np.array(given, np.float32)
            logstd = np.array(logstd, np.float32)
            std = np.exp(logstd)
            target_log_p = np.array(stats.norm.logpdf(given, mean, np.exp(logstd)), np.float32)
            target_p = np.array(stats.norm.pdf(given, mean, np.exp(logstd)), np.float32)

            mean = torch.tensor(mean)
            logstd = torch.tensor(logstd)
            std = torch.tensor(std)
            given = torch.tensor(given)
            norm1 = Normal(mean=mean, logstd=logstd)
            log_p1 = norm1.log_prob(given)
            np.testing.assert_allclose(log_p1.numpy(), target_log_p, rtol=1e-03)

            # TODO: May add prob function to Normal module in the future
            # p1 = norm1.prob(given)
            # np.testing.assert_allclose(p1.numpy(), target_p)
            # # self.assertAllClose(p1.eval(), target_p)

            norm2 = Normal(mean=mean, std=std)
            log_p2 = norm2.log_prob(given)
            np.testing.assert_allclose(log_p2.numpy(), target_log_p, rtol=1e-03)

            # p2 = norm2.prob(given)
            # np.testing.assert_allclose(p2.numpy(), target_p)
            # # self.assertAllClose(p2.eval(), target_p)

        # TODO: Edit Normal distribution module to support integer inputs
        # _test_value(0., 0., 0.)
        _test_value([0.], [0.], [0.])
        _test_value([0.99, 0.9, 9., 99.], [1.], [-3., -1., 1., 10.])
        _test_value([7.], [0., 4.], [[1., 2.], [3., 5.]])

    def test_check_numerics(self):
        norm1 = Normal(mean=torch.ones([1, 2]),
                       logstd=torch.tensor([[-1e10]]),
                       check_numerics=True)
        try:
            norm1.log_prob(torch.tensor([0.])).numpy()
        except:
            raise AttributeError("precision.*Tensor had Inf")

        norm2 = Normal(mean=torch.ones([1, 2]),
                       logstd=torch.tensor([[1e3]]),
                       check_numerics=True)
        try:
            norm2.sample().numpy()
        except:
            raise AttributeError("exp(logstd).*Tensor had Inf")

        norm3 = Normal(mean=torch.ones([1, 2]),
                       std=torch.tensor([[0.]]),
                       check_numerics=True)
        try:
            norm3.log_prob(torch.tensor([0.])).numpy()
        except:
            raise AttributeError("log(std).*Tensor had Inf")

    def test_dtype(self):
        utils.test_dtype_2parameter(self, self._Normal_std)
        utils.test_dtype_2parameter(self, self._Normal_logstd)

    def test_distribution_shape(self):
        param1 = torch.zeros([1])
        param2 = torch.ones([1])
        distribution = self._Normal_logstd(param1, param2)
        utils.test_and_save_distribution_img(distribution)


if __name__ == '__main__':
    unittest.main()
