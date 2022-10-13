import torch

from zhusuan.distributions import Distribution

__all__ = [
    "FlowDistribution",
]


class FlowDistribution(Distribution):
    """
    A class for sample from Flow networks by provide the latent distribution and the flow network,
    when calling `sample` method, it returns the sample from flow network, when calling `log_prob` method
    it return the loss item of flow network.

    :param latents: An instance of `Distribution` class, as the prior（or the latent variable）of FlowDistrubution
    :param transformation: A RevNet instance, the Flow net work built by user
    :param flow_kwargs: additional info to be recode
    :param dtype: data type
    :param device: device of Distribution

    """

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
        else:  # if n_samples == -1, then return None as no sample
            return None
        x, _ = self._transformation.forward(z, reverse=True, **kwargs)
        return x

    def _log_prob(self, *given, **kwargs):
        z, log_det_J = self._transformation.forward(*given, **kwargs, reverse=False)
        log_ll = torch.sum(self._latents.log_prob(z), dim=1)
        return log_ll + log_det_J
