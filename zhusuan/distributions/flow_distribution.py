import torch

from .base import Distribution

__all__ = [
    "FlowDistribution",
]


class FlowDistribution(Distribution):
    def __init__(self, latents=None, transformation=None, flow_kwargs=None, dtype=torch.float32, group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):
        self._latents = latents
        self._transformation = transformation
        self._flow_kwargs = flow_kwargs

        super(FlowDistribution, self).__init__(
            dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            device=device,
            **kwargs
        )

    def _sample(self, n_samples=-1, shape=None, **kwargs):
        if shape is not None:
            z = self._latents.sample(shape=shape)
        elif n_samples != -1:
            z = self._latents.sample(n_samples)
        else:
            z = 0.
        x, _ = self._transformation.forward(z, reverse=True, shape=shape, **kwargs)
        return x[0]

    def log_prob(self, *given, **kwargs):
        z, log_det_J = self._transformation.forward(*given, **kwargs, reverse=False)
        log_ll = torch.sum(self._latents.log_prob(z[0]) + log_det_J, dim=1)
        return log_ll
