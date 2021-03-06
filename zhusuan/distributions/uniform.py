import torch
from zhusuan.distributions import Distribution

class Uniform(Distribution):
    """
    The class of univariate Uniform distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param low: A 'float' Var. Lower range (inclusive).
    :param high: A 'float' Var. Upper range (exclusive).
    """
    def __init__(self,
                dtype=torch.float32,
                is_continues=True,
                is_reparameterized=True,
                group_ndims=0,
                device=torch.device('cpu'),
                **kwargs):
        super(Uniform, self).__init__(dtype,
                                       is_continues,
                                       is_reparameterized,
                                       group_ndims=group_ndims,
                                       device=device,
                                       **kwargs)

        self._low = torch.as_tensor(kwargs['low'], dtype = self._dtype).to(device) if type(kwargs['low']) in [int, float] else kwargs['low'].to(device)
        self._high = torch.as_tensor(kwargs['high'], dtype = self._dtype).to(device) if type(kwargs['high']) in [int, float] else kwargs['high'].to(device)

    @property
    def low(self):
        """Lower range (inclusive) of the Uniform distribution."""
        return self._low

    @property
    def high(self):
        """Upper range (exclusive) of the Uniform distribution."""
        return self._high

    def _sample(self, n_samples=1):
        if n_samples > 1:
            _shape = self._low.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._low.shape)
            _low = self._low.repeat([n_samples, *_len * [1]])
            _high = self._high.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._low.shape
            _low = torch.as_tensor(self._low, dtype=self._dtype)
            _high = torch.as_tensor(self._high, dtype=self._dtype)

        if not self.is_reparameterized:
            _sample = torch.distributions.uniform.Uniform(_low, _high).sample()  
        else:
            _sample = torch.distributions.uniform.Uniform(torch.zeros(_shape, dtype=self._dtype), torch.ones(_shape, dtype=self._dtype)).sample()
        self.sample_cache = _sample
        return _sample*(_high-_low)+_low

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._low.shape):
            n_samples = sample.shape[0]
            _len = len(self._low.shape)
            _low = self._low.repeat([n_samples, *_len * [1]])
            _high = self._high.repeat([n_samples, *_len * [1]])
        else:
            _low = self._low
            _high = self._high

        return torch.distributions.uniform.Uniform(_low, _high).log_prob(sample)