import torch
from zhusuan.distributions import Distribution
from zhusuan.distributions.utils import(
    assert_same_log_float_dtype,
    check_broadcast
)


class Gamma(Distribution):
    """
    The class of univariate Gamma distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param alpha: A 'float' Var. Shape parameter of the Gamma distribution.
    :param beta: A 'float' Var. Rate parameter of the Gamma distribution.
    """

    def __init__(self,
                 alpha,
                 beta,
                 dtype=None,
                 is_continuous=True,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):
        self._alpha = torch.as_tensor(alpha, dtype=dtype).to(device)
        self._beta = torch.as_tensor(beta, dtype=dtype).to(device)
        check_broadcast(self.alpha, self.beta)
        dtype = assert_same_log_float_dtype([(self._alpha, "Gamma.alpha"), (self._beta, "Gamma.beta")])
        super(Gamma, self).__init__(dtype,
                                    is_continuous,
                                    is_reparameterized=False,
                                    # reparameterization trick is not applied for Gamma distribution
                                    group_ndims=group_ndims,
                                    device=device,
                                    **kwargs)

    @property
    def alpha(self):
        """Shape parameter of the Gamma distribution."""
        return self._alpha

    @property
    def beta(self):
        """Rate parameter of the Gamma distribution."""
        return self._beta

    def _batch_shape(self):
        return torch.broadcast_shapes(self.alpha.shape, self.beta.shape)

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
        self.sample_cache = _sample
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

    def _prob(self, given):
        return torch.exp(self._log_prob(given))