import torch
from zhusuan.distributions import Distribution

class Gamma(Distribution):
    """
    The class of univariate Gamma distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param alpha: A 'float' Var. Shape parameter of the Gamma distribution.
    :param beta: A 'float' Var. Rate parameter of the Gamma distribution.
    """
    def __init__(self,
                dtype=torch.float32,
                is_continues=True,
                group_ndims=0,
                device=torch.device('cpu'),
                **kwargs):
        super(Gamma, self).__init__(dtype,
                                    is_continues,
                                    is_reparameterized=False, # reparameterization trick is not applied for Gamma distribution
                                    group_ndims=group_ndims,
                                    device=device,
                                    **kwargs)

        self._alpha = torch.as_tensor(kwargs['alpha'], dtype = self._dtype).to(device) if type(kwargs['alpha']) in [int, float] else kwargs['alpha'].to(device)
        self._beta = torch.as_tensor(kwargs['beta'], dtype = self._dtype).to(device) if type(kwargs['beta']) in [int, float] else kwargs['beta'].to(device)

    @property
    def alpha(self):
        """Shape parameter of the Gamma distribution."""
        return self._alpha

    @property
    def beta(self):
        """Rate parameter of the Gamma distribution."""
        return self._beta

    def _sample(self, n_samples=1):
        if n_samples > 1:
            _shape = self._alpha.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._alpha.shape)
            _alpha = self._alpha.repeat([n_samples, *_len * [1]])
            _beta = self._beta.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._alpha.shape
            _alpha = torch.as_tensor(self._alpha, dtype=self._dtype)
            _beta = torch.as_tensor(self._beta, dtype=self._dtype)

        _sample = torch.distributions.gamma.Gamma(_alpha, _beta).sample()

        return _sample

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._alpha.shape):
            n_samples = sample.shape[0]
            _len = len(self._alpha.shape)
            _alpha = self._alpha.repeat([n_samples, *_len * [1]])
            _beta = self._beta.repeat([n_samples, *_len * [1]])
        else:
            _alpha = self._alpha
            _beta = self._beta

        return torch.distributions.gamma.Gamma(_alpha, _beta).log_prob(sample)