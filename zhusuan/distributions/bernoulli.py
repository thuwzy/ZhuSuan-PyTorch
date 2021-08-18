import torch
import numpy as np

from zhusuan.distributions.base import Distribution

class Bernoulli(Distribution):
    """
    The class of univariate Bernoulli distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A `float` Tensor. The log-odds of probabilities of being 1.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

    :param dtype: The value type of samples from the distribution. Can be
        int (`torch.int16`, `torch.int32`, `torch.int64`) or float (`torch.float16`,
        `torch.float32`, `torch.float64`). Default is `int32`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    """
    def __init__(self,
                 dtype=torch.float32,
                 param_dtype=torch.float32,
                 is_continues=False,
                 is_reparameterized=True,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):
        super(Bernoulli, self).__init__(dtype,
                                        param_dtype,
                                        is_continues,
                                        is_reparameterized,
                                        group_ndims=group_ndims,
                                        device=device,
                                        **kwargs)
        self._probs = kwargs['probs']
        self._probs = torch.as_tensor(self._probs, dtype=self._dtype)

    @property
    def probs(self):
        return self._probs

    def _batch_shape(self):
        return self.probs.shape

    def _sample(self, n_samples=1):
        if n_samples > 1:
            sample_shape = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
            _probs = self._probs * torch.ones(sample_shape)
        else:
            _probs = self._probs# * torch.ones(self.batch_shape)

        #_probs *= torch.tensor(_probs <= 1, dtype=self._dtype) #! Values larger than 1 are set to 0
        _sample = torch.bernoulli(_probs)
        _sample = torch.as_tensor(_sample, dtype=self._dtype)

        self.sample_cache = _sample
        return _sample

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        if len(sample.shape) > len(self._probs.shape):
            sample_shape = np.concatenate([[sample.shape[0]], self.batch_shape], axis=0).tolist()
            _probs = self._probs * torch.ones(sample_shape)
        else:
            _probs = self._probs# * torch.ones(self.batch_shape)

        log_prob = sample * torch.log(_probs + 1e-8) + (1 - sample) * torch.log(1 - _probs + 1e-8)
        return log_prob
        #! Check it again