#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np

import matplotlib.pyplot as plt
import os

__all__ = [
    'test_and_save_distribution_img',
    'test_dtype_1parameter_discrete',
    'test_dtype_2parameter',
    'test_1parameter_log_prob_shape_same',
    'test_2parameter_log_prob_shape_same',
    'test_1parameter_sample_shape_same',
    'test_2parameter_sample_shape_same',
    'test_batch_shape_1parameter',
    'test_batch_shape_2parameter_univariate'
]

def test_and_save_distribution_img( distribution,
                                    hist_folder=os.path.join(
                                        os.path.dirname(__file__),'hist_images')):
    # Test sample hist and save image to histogram folder
    if not os.path.isdir(hist_folder):
        os.mkdir(hist_folder)

    samples = distribution.sample(10000).numpy().flatten()

    dist_name = distribution.__class__.__name__
    img_path = os.path.join(hist_folder, dist_name+'.png')


    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))
    ax0.hist(samples, 200,  histtype='bar', facecolor='blue', alpha=0.75)
    ## Draw pdf
    ax0.set_title('pdf')
    ax1.hist(samples, 100,  histtype='bar', facecolor='pink',
             alpha=0.75, cumulative=True, rwidth=0.8)
    # Draw cdf
    ax1.set_title("cdf")
    fig.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig(img_path)


def test_dtype_1parameter_discrete(
        test_class, Distribution, prob_only=False, allow_16bit=True):

    def _test_sample_dtype(input_, result_dtype, **kwargs):
        distribution = Distribution(input_, **kwargs)
        samples = distribution.sample(2)
        # test_class.assertEqual(distribution.dtype, result_dtype)
        test_class.assertEqual(samples.dtype, result_dtype)

    def _test_sample_dtype_raise(input_, dtype):
        try:
            _ = Distribution(input_, dtype=dtype)
        except:
            raise TypeError("`dtype`.*not in")
    
    if not prob_only:
        for input_elem in [[1.], [[2., 3.], [4., 5.]]]:
            input_ = torch.tensor(input_elem)
            _test_sample_dtype(input_, torch.int32, dtype=torch.int32)

            if allow_16bit:
                _test_sample_dtype(input_, torch.int16, dtype=torch.int16)
                _test_sample_dtype(input_, torch.float16, dtype=torch.float16)
            else:
                _test_sample_dtype_raise(input_, dtype=torch.int16)
                _test_sample_dtype_raise(input_, dtype=torch.float16)

            _test_sample_dtype(input_, torch.int32, dtype=torch.int32)
            _test_sample_dtype(input_, torch.int64, dtype=torch.int64)
            _test_sample_dtype(input_, torch.float32, dtype=torch.float32)
            _test_sample_dtype(input_, torch.float64, dtype=torch.float64)
            _test_sample_dtype_raise(input_, dtype=torch.uint8)
            _test_sample_dtype_raise(input_, dtype=torch.bool)


def test_dtype_2parameter(test_class, Distribution):
    # Test sample dtype
    def _test_sample_dtype(dtype):
        param1 = torch.zeros([1], dtype=dtype)
        param2 = torch.ones([1], dtype=dtype)
        distribution = Distribution(param1, param2)
        test_class.assertEqual(dtype, distribution.sample(1).dtype)

    _test_sample_dtype(torch.float16)
    _test_sample_dtype(torch.float32)
    _test_sample_dtype(torch.float64)

    # Test log_prob and prob dtype
    def _test_log_prob_dtype(dtype):
        param1 = torch.zeros([1], dtype=dtype)
        param2 = torch.ones([1], dtype=dtype)
        distribution = Distribution(param1, param2)

        # test for tensor
        given = torch.zeros([1], dtype=dtype)
        # test_class.assertEqual(distribution.prob(given).dtype, dtype)
        test_class.assertEqual(distribution.log_prob(given).dtype, dtype)

        # # test for numpy
        # given_np = torch.tensor(given.numpy())
        # # test_class.assertEqual(distribution.prob(given_np).dtype, dtype)
        # test_class.assertEqual(distribution.log_prob(given_np).dtype, dtype)

    _test_log_prob_dtype(torch.float16)
    _test_log_prob_dtype(torch.float32)
    _test_log_prob_dtype(torch.float64)

    # Test dtype for parameters
    def _test_parameter_dtype(result_dtype, param1_dtype, param2_dtype):
        param1 = torch.zeros([1], dtype=param1_dtype)
        param2 = torch.ones([1], dtype=param2_dtype)
        distribution = Distribution(param1, param2)
        test_class.assertEqual(distribution.sample().dtype, result_dtype)


    _test_parameter_dtype(torch.float16, torch.float16, torch.float16)
    _test_parameter_dtype(torch.float32, torch.float32, torch.float32)
    _test_parameter_dtype(torch.float64, torch.float64, torch.float64)

    # def _test_parameter_dtype_raise(param1_dtype, param2_dtype):
    #     if param1_dtype != param2_dtype:
    #         regexp_msg = "must have the same dtype as"
    #     else:
    #         regexp_msg = "must have a dtype in"
    #     try:
    #         _test_parameter_dtype(None, param1_dtype, param2_dtype)
    #     except:
    #         raise TypeError(regexp_msg)
    #
    # _test_parameter_dtype_raise(torch.float16, torch.float32)
    # _test_parameter_dtype_raise(torch.float32, torch.float64)
    # _test_parameter_dtype_raise(torch.int32, torch.int32)



def test_1parameter_log_prob_shape_same(
        test_class, Distribution, make_param, make_given):

    def _test_dynamic(param_shape, given_shape, target_shape):

        param = torch.tensor(make_param(param_shape), dtype=torch.float32)
        dist = Distribution(param)

        given = torch.tensor(make_given(given_shape), dtype=torch.float32)
        log_p = dist.log_prob(given)
        test_class.assertEqual(log_p.shape, target_shape)

    _test_dynamic([2, 3], [1, 3], [2, 3])
    _test_dynamic([1, 3], [2, 2, 3], [2, 2, 3])
    _test_dynamic([1, 5], [1, 2, 3, 1], [1, 2, 3, 5])
    # try:
    #     _test_dynamic([2, 3, 5], [1, 2, 1], None)
    # except:
    #     raise AssertionError("Incompatible shapes")
    #


def test_2parameter_log_prob_shape_same(
        test_class, Distribution, make_param1, make_param2, make_given):

    def _test_dynamic(param1_shape, param2_shape, given_shape,
                      target_shape):
        param1 = torch.tensor(make_param1(param1_shape), dtype=torch.float32)
        param2 = torch.tensor(make_param2(param2_shape), dtype=torch.float32)
        dist = Distribution(param1, param2)
        given = torch.tensor(make_given(given_shape), dtype=torch.float32)
        log_p = dist.log_prob(given)
        test_class.assertEqual( log_p.shape, torch.Size(target_shape))

    _test_dynamic([2, 3], [2, 1], [1, 3], [2, 3])
    _test_dynamic([1, 3], [1, 1], [2, 1, 3], [2, 1, 3])
    _test_dynamic([1, 5], [3, 1], [1, 2, 1, 1], [1, 2, 3, 5])
    # try:
    #     _test_dynamic([2, 3, 5], [1, 1, 1], [1, 2, 1], None)
    # except:
    #     raise AssertionError("Incompatible shapes")



def test_1parameter_sample_shape_same(
        test_class, Distribution, make_param, only_one_sample=False):

    def _test_dynamic(param_shape, n_samples, target_shape):
        param = torch.tensor(make_param(param_shape), dtype=torch.float32)
        dist = Distribution(param)
        samples = dist.sample(n_samples)
        test_class.assertEqual(samples.shape, target_shape)

    _test_dynamic([2, 3], 1, [2, 3])
    if not only_one_sample:
        _test_dynamic([1, 3], 2, [2, 1, 3])
        _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])


def test_2parameter_sample_shape_same(
        test_class, Distribution, make_param1, make_param2):

    def _test_dynamic(param1_shape, param2_shape, n_samples,
                      target_shape):
        param1 = torch.tensor(make_param1(param1_shape), dtype=torch.float32)
        param2 = torch.tensor(make_param2(param2_shape), dtype=torch.float32)
        dist = Distribution(param1, param2)
        samples = dist.sample(n_samples)
        test_class.assertEqual(samples.shape, torch.Size(target_shape))

    # TODO: Sample_shape will be [n_samples, batch_shape]
    _test_dynamic([2, 3], [2, 1], 1, [2, 3])
    _test_dynamic([1, 3], [2, 1], 2, [2, 2, 3])
    _test_dynamic([2, 1, 5], [1, 3, 1], 3, [3, 2, 3, 5])
    # try:
    #     _test_dynamic([2, 3, 5], [1, 1, 1], 1, None)
    # except:
    #     raise AssertionError("Incompatible shapes")



def test_batch_shape_1parameter(
        test_class, Distribution, make_param, is_univariate):

    # dynamic
    def _test_dynamic(param_shape):
        param = torch.tensor(make_param(param_shape), dtype=torch.float32)
        dist = Distribution(param)
        test_class.assertEqual(dist.batch_shape,
                               param_shape if is_univariate else param_shape[:-1])
    if is_univariate:
        _test_dynamic([])
    _test_dynamic([2])
    _test_dynamic([2, 3])
    _test_dynamic([2, 1, 4])


def test_batch_shape_2parameter_univariate(
        test_class, Distribution, make_param1, make_param2):

    # dynamic
    def _test_dynamic(param1_shape, param2_shape, target_shape):
        param1 = torch.tensor(make_param1(param1_shape), dtype=torch.float32)
        param2 = torch.tensor(make_param2(param2_shape), dtype=torch.float32)
        dist = Distribution(param1, param2)
        # test_class.assertTrue(np.array(dist.batch_shape).dtype is np.int32)
        test_class.assertEqual( dist.batch_shape, target_shape)

    # _test_dynamic([2, 3], [], [2, 3])
    _test_dynamic([2, 3], [3], [2, 3])
    _test_dynamic([2, 1, 4], [2, 3, 4], [2, 3, 4])
    _test_dynamic([2, 3, 5], [3, 1], [2, 3, 5])
    # try:
    #     _test_dynamic([2, 3, 5], [3, 2], None)
    # except:
    #     AssertionError("Incompatible shapes")


