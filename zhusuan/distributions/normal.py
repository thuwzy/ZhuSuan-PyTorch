# -*- coding: utf-8 -*-
import torch
import numpy as np

from zhusuan.distributions.base import Distribution
from zhusuan.distributions.utils import (
    assert_same_float_dtype,
    assert_same_log_float_dtype,
)


class Normal(Distribution):
    """
    The class of univariate Normal distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param mean: A `float` Var. The mean of the Normal distribution.
        Should be broadcastable to match `std` or `logstd`.
    :param std: A `float` Var. The standard deviation of the Normal
        distribution. Should be positive and broadcastable to match `mean`.
    :param logstd: A `float` Var. The log standard deviation of the Normal
        distribution. Should be broadcastable to match `mean`.
    :param group_ndims: A 0-D `int32` Var representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param use_path_derivative: A bool. Whether when taking the gradients
        of the log-probability to propagate them through the parameters
        of the distribution (False meaning you do propagate them). This
        is based on the paper "Sticking the Landing: Simple,
        Lower-Variance Gradient Estimators for Variational Inference"
    """

    def __init__(self,
                 mean=0.,
                 std=None,
                 logstd=None,
                 dtype=None,
                 is_continuous=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):
        self._mean: torch.Tensor = torch.as_tensor(mean, dtype=dtype).to(device)
        if (logstd is None) == (std is None):
            raise ValueError(
                "Either `std` or `logstd` should be passed. It is not allowed "
                "that both are specified or both are not.")
        elif std is None:
            self._std: torch.Tensor = torch.exp(torch.as_tensor(logstd, dtype=dtype)).to(device)
        else:  # logstd is None:
            self._std: torch.Tensor = torch.as_tensor(std, dtype=dtype).to(device)

        # check dtype:
        dtype = assert_same_log_float_dtype([(self._mean, "Normal.mean"), (self._std, "Normal.std")])

        super(Normal, self).__init__(dtype=dtype,
                                     is_continuous=is_continuous,
                                     is_reparameterized=is_reparameterized,
                                     group_ndims=group_ndims,
                                     device=device,
                                     **kwargs)

    @property
    def mean(self):
        """The mean of the Normal distribution."""
        return self._mean

    @property
    def logstd(self):
        """The log standard deviation of the Normal distribution."""
        return torch.log(self._std)

    @property
    def std(self):
        """The standard deviation of the Normal distribution."""
        return self._std

    def _batch_shape(self):
        return torch.broadcast_shapes(self.mean.shape, self.std.shape)

    def _sample(self, n_samples=1):
        if n_samples > 1:
            _shape = self._mean.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._mean.shape)
            _mean = self._mean.repeat([n_samples, *_len * [1]])
            _std = self._std.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._mean.shape
            _mean = self._mean
            _std = self._std

        if not self.is_reparameterized:
            _sample = torch.normal(_mean, _std).to(self.device)
        else:
            epsilon = torch.normal(0., 1., size=_shape).to(self.device)
            _sample = _mean + _std * epsilon
        self.sample_cache = _sample
        return _sample

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._mean.shape):
            n_samples = sample.shape[0]
            _len = len(self._mean.shape)
            _mean = self._mean.repeat([n_samples, *_len * [1]])
            _std = self._std.repeat([n_samples, *_len * [1]])
        else:
            _mean = self._mean
            _std = self._std

        logstd = torch.log(_std).to(self.device)
        c = torch.tensor(-0.5 * np.log(2 * np.pi)).to(self.device)
        precision = torch.exp(-2 * logstd)
        # precision = torch.where(torch.isinf(precision), torch.full_like(precision, 1e10), precision)
        log_prob = c - logstd - 0.5 * precision * ((sample - _mean) ** 2)

        def print_m(t):
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    print("{:.2f} ,".format(t[i][j]), end="")
                print()

        if torch.any(torch.isnan(log_prob)):
            if torch.any(torch.isnan(precision)):
                print("preci")
            elif torch.any(torch.isnan(logstd)):
                print("std")
            elif torch.any(torch.isnan(((sample - _mean) ** 2))):
                print(sample)
                print(_mean)
                print("????")
            elif torch.any(torch.isnan(precision * ((sample - _mean) ** 2))):
                print("??")
                zeo = ((sample - _mean) ** 2)
                ans = precision * zeo
                for index1 in range(ans.shape[0]):
                    if torch.any(torch.isnan(ans[index1])):
                        for index2 in range(ans.shape[1]):
                            if torch.any(torch.isnan(ans[index1][index2])):
                                print(ans[index1][index2])
                                print(precision[index1][index2])
                                print(zeo[index1][index2])
                                break
            exit(1)
        return log_prob

    def _prob(self, given):
        return torch.exp(self._log_prob(given))
