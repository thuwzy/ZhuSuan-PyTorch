import torch

from zhusuan.distributions import Distribution

__all__ = [
    "FlowDistribution",
]


class FlowDistribution(Distribution):
    def __init__(self, latents, transformation, flow_kwargs=None, dtype=torch.float32, group_ndims=0,
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

    def _sample(self, n_samples=-1, **kwargs):
        if n_samples != -1:
            z = self._latents.sample(n_samples)
        else: # if n_samples == -1, then return None as no sample
            return None
        x, _ = self._transformation.forward(z, reverse=True, **kwargs)
        return x

    def log_prob(self, *given, **kwargs):
        z, log_det_J = self._transformation.forward(*given, **kwargs, reverse=False)
        log_ll = torch.sum(self._latents.log_prob(z), dim=1)
        return log_ll + log_det_J
