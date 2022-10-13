import torch
from zhusuan.distributions import Distribution
from zhusuan.distributions.utils import (
    assert_same_log_float_dtype
)


class Exponential(Distribution):
    """
    The class of univariate Exponential distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param rate: A 'float' Var. Rate parameter of the Exponential distribution.
    """

    def __init__(self,
                 rate,
                 dtype=None,
                 is_continues=True,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):
        self._rate = torch.as_tensor(rate, dtype=dtype).to(device)
        dtype = assert_same_log_float_dtype([(self._rate, "Exponential.rate")])
        super(Exponential, self).__init__(dtype,
                                          is_continues,
                                          is_reparameterized=False,
                                          # reparameterization trick is not applied for Exponential distribution
                                          group_ndims=group_ndims,
                                          device=device,
                                          **kwargs)

    @property
    def rate(self):
        """Shape parameter of the Exponential distribution."""
        return self._rate

    def _batch_shape(self):
        return self._rate.shape

    def _sample(self, n_samples=1):
        if n_samples > 1:
            _shape = self._rate.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._rate.shape)
            _rate = self._rate.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._rate.shape
            _rate = torch.as_tensor(self._rate, dtype=self._dtype)

        _sample = torch.distributions.exponential.Exponential(_rate).sample()
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

        return torch.distributions.exponential.Exponential(_rate).log_prob(sample)
