import torch
import warnings
from zhusuan.distributions import Distribution
from zhusuan.distributions.utils import (
    assert_same_log_float_dtype,
    integer_dtypes,
    int2float_mapping
)

class Poisson(Distribution):
    """
    The class of univariate Poisson distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param rate: A 'float' Var. Rate parameter of the Poisson distribution.Must be positive.

    """

    def __init__(self,
                 rate,
                 dtype=None,
                 is_continues=True,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):

        self._rate = torch.as_tensor(rate, dtype=dtype).to(device)
        if self._rate.dtype in integer_dtypes:
            warnings.warn(f"the tensor dtype convert {self._rate.dtype} to  {int2float_mapping[self._rate.dtype]}")
            self._rate = torch.as_tensor(self._rate, dtype=int2float_mapping[self._rate.dtype])

        if dtype is None:
            dtype = assert_same_log_float_dtype([(self._rate, "Poisson.mean")])

        super(Poisson, self).__init__(dtype,
                                      is_continues,
                                      is_reparameterized=False,
                                      # reparameterization trick is not applied for Poisson distribution
                                      group_ndims=group_ndims,
                                      device=device,
                                      **kwargs)

    @property
    def rate(self):
        """Shape parameter of the Poisson distribution."""
        return self._rate

    def _sample(self, n_samples=1):
        if n_samples > 1:
            _shape = self._rate.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._rate.shape)
            _rate = self._rate.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._rate.shape
            _rate = torch.as_tensor(self._rate, dtype=self._dtype)

        _sample = torch.distributions.poisson.Poisson(_rate).sample()
        self.sample_cache = _sample
        return _sample

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._rate.shape):
            n_samples = sample.shape[0]
            _len = len(self._rate.shape)
            _rate = self._rate.repeat([n_samples, *_len * [1]])
        else:
            _rate = self._rate

        return torch.distributions.poisson.Poisson(_rate).log_prob(sample)

    def _prob(self, given):
        return torch.exp(self._log_prob(given))
