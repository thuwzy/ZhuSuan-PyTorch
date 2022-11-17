import torch
from zhusuan.distributions import Distribution
from zhusuan.distributions.utils import (
    assert_same_log_float_dtype,
    check_broadcast
)


class Laplace(Distribution):
    """
    The class of univariate Laplace distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param loc: A 'float' Var. Mean of the Laplace distribution.
    :param scale: A 'float' Var. Scale of the Laplace distribution.
    """

    def __init__(self,
                 loc,
                 scale,
                 dtype=None,
                 is_continuous=True,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):
        self._loc = torch.as_tensor(loc, dtype=dtype).to(device)
        self._scale = torch.as_tensor(scale, dtype=dtype).to(device)
        check_broadcast(self.loc, self.scale)
        dtype = assert_same_log_float_dtype([(self._loc, "Laplace.loc"), (self._scale, "Laplace.scale")])
        super(Laplace, self).__init__(dtype,
                                      is_continuous,
                                      is_reparameterized=False,
                                      # reparameterization trick is not applied for Laplace distribution
                                      group_ndims=group_ndims,
                                      device=device,
                                      **kwargs)

    @property
    def loc(self):
        """Mean of the Laplace distribution."""
        return self._loc

    @property
    def scale(self):
        """Scale of the Laplace distribution."""
        return self._scale

    def _batch_shape(self):
        return torch.broadcast_shapes(self.loc.shape, self.scale.shape)

    def _sample(self, n_samples=1):
        if n_samples > 1:
            _shape = self._loc.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._loc.shape)
            _loc = self._loc.repeat([n_samples, *_len * [1]])
            _scale = self._scale.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._loc.shape
            _loc = torch.as_tensor(self._loc, dtype=self._dtype)
            _scale = torch.as_tensor(self._scale, dtype=self._dtype)

        _sample = torch.distributions.laplace.Laplace(_loc, _scale).sample()
        self.sample_cache = _sample
        return _sample

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._loc.shape):
            n_samples = sample.shape[0]
            _len = len(self._loc.shape)
            _loc = self._loc.repeat([n_samples, *_len * [1]])
            _scale = self._scale.repeat([n_samples, *_len * [1]])
        else:
            _loc = self._loc
            _scale = self._scale

        return torch.distributions.laplace.Laplace(_loc, _scale).log_prob(sample)

    def _prob(self, given):
        return torch.exp(self._log_prob(given))