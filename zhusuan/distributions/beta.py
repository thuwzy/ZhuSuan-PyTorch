import torch
import numpy as np

from zhusuan.distributions import Distribution

class Beta(Distribution):
    """
    The class of univariate Beta distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param alpha: A 'float' Var. One of the two shape parameters of the Beta distribution.
    :param beta: A 'float' Var. One of the two shape parameters of the Beta distribution.
    :param is_reparameterized: A Bool. If True, gradients on samples from this distribution are allowed to propagate into inputs, using the reparametrization trick from (Kingma, 2013).
    """
    def __init__(self,
                dtype=torch.float32,
                param_dtype=torch.float32,
                is_continues=True,
                is_reparameterized=True,
                group_ndims=0,
                device=torch.device('cpu'),
                **kwargs):
        super(Beta, self).__init__(dtype,
                                       param_dtype,
                                       is_continues,
                                       is_reparameterized,
                                       group_ndims=group_ndims,
                                       device=device,
                                       **kwargs)

        self._alpha = torch.as_tensor(kwargs['alpha'], dtype = self._dtype).to(device) if type(kwargs['alpha']) in [int, float] else kwargs['alpha'].to(device)
        self._beta = torch.as_tensor(kwargs['beta'], dtype = self._dtype).to(device) if type(kwargs['beta']) in [int, float] else kwargs['beta'].to(device)

    @property
    def alpha(self):
        """One of the two shape parameters of the Beta distribution."""
        return self._alpha

    @property
    def beta(self):
        """One of the two shape parameters of the Beta distribution."""
        return self._beta   