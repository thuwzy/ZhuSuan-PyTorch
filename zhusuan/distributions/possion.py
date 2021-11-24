import torch
from zhusuan.distributions import Distribution

class Possion(Distribution):
    """
    The class of univariate Possion distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param rate: A 'float' Var. Rate parameter of the Possion distribution.
    """
    def __init__(self,
                dtype=torch.float32,
                is_continues=True,
                group_ndims=0,
                device=torch.device('cpu'),
                **kwargs):
        super(Possion, self).__init__(dtype,
                                       is_continues,
                                       is_reparameterized=False, # reparameterization trick is not applied for Possion distribution
                                       group_ndims=group_ndims,
                                       device=device,
                                       **kwargs)

        self._rate = torch.as_tensor(kwargs['rate'], dtype = self._dtype).to(device) if type(kwargs['rate']) in [int, float] else kwargs['rate'].to(device)

    @property
    def rate(self):
        """Shape parameter of the Possion distribution."""
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