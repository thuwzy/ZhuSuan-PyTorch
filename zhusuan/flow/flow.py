import torch

from zhusuan.distributions import Distribution

class Flow(Distribution):
    """
    A abstract distribution class which represents the transformed distribution.
    It contains a known latent distribution and a list of transforms. The `log_prob` method use forward
    transforms to transform data to noise and compute the log likelihood. The `sample` method can transform 
    a noise in latent space to a data. The transform process is:

    .. math::
    
        p_ {\\theta}(x) = p_{\\theta}(z) \\left\\vert \det{\\left(\\frac{\partial f^{-1}}{\partial x}\\right)}\\right\\vert

    :param latent: A instance of :class:`~zhusuan.distributions.base.Distribution`, the prior distribution of normalizing flows.
    :param transform: The invertible transforms which will apply to the data. Typicially is a instance of :class:`~zhusuan.transforms.sequential.Sequential` or a single :class:`~zhusuan.transforms.invertible.base.InvertibleTransform`.
    """
    def __init__(self, latent=None, transform=None, dtype='float32', group_ndims=0, **kwargs):
        super().__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

        self._latent = latent
        self._transform = transform
    
    def _sample(self, n_samples=1, **kwargs):
        if n_samples == -1:
            return 0
        else:
            z = self._latent.sample(n_samples)
            x_hat = self._transform.forward(z, inverse=True, **kwargs)
            return x_hat[0]

    def _log_prob(self, *given, **kwargs):
        z, log_detJ = self._transform.forward(*given, inverse=False, **kwargs)
        log_likelihood = torch.sum(self._latent.log_prob(z[0]) + log_detJ, 1)
        return log_likelihood
