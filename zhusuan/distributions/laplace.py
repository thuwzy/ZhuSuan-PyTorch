import torch
from zhusuan.distributions import Distribution

class Laplace(Distribution):
    """
    The class of univariate Laplace distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param loc: A 'float' Var. Mean of the Laplace distribution.
    :param scale: A 'float' Var. Scale of the Laplace distribution.
    """
    def __init__(self,
                dtype=torch.float32,
                param_dtype=torch.float32,
                is_continues=True,
                group_ndims=0,
                device=torch.device('cpu'),
                **kwargs):
        super(Laplace, self).__init__(dtype,
                                       param_dtype,
                                       is_continues,
                                       is_reparameterized=False, # reparameterization trick is not applied for Laplace distribution
                                       group_ndims=group_ndims,
                                       device=device,
                                       **kwargs)

        self._loc = torch.as_tensor(kwargs['loc'], dtype = self._dtype).to(device) if type(kwargs['loc']) in [int, float] else kwargs['loc'].to(device)
        self._scale = torch.as_tensor(kwargs['scale'], dtype = self._dtype).to(device) if type(kwargs['scale']) in [int, float] else kwargs['scale'].to(device)

    @property
    def loc(self):
        """Mean of the Laplace distribution."""
        return self._loc

    @property
    def scale(self):
        """Scale of the Laplace distribution."""
        return self._scale

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