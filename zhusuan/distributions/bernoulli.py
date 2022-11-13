import torch
import numpy as np

from zhusuan.distributions.base import Distribution
from zhusuan.distributions.utils import (
    assert_same_log_float_dtype
)


class Bernoulli(Distribution):
    """
    The class of univariate Bernoulli distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A `float` Tensor. The log-odds of probabilities of being 1.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}
    :param probs: A 'float' Tensor. The p param of bernoulli distribution

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
                 logits=None,
                 probs=None,
                 dtype=None,
                 is_continues=False,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):
        if (logits is None) == (probs is None):
            raise ValueError(
                "Either `probs` or `logits` should be passed. It is not allowed "
                "that both are specified or both are not.")
        elif logits is None:
            self._probs: torch.Tensor = torch.as_tensor(probs, dtype=dtype).to(device)
        else:  # probs is None
            _logits = torch.as_tensor(logits, dtype=dtype)
            assert_same_log_float_dtype([(_logits, "Bernoulli.logits")])
            self._probs: torch.Tensor = torch.sigmoid(_logits).to(device)
        # dtype of probs must be float32 or float64
        dtype = assert_same_log_float_dtype([(self._probs, "Bernoulli.probs")])
        super(Bernoulli, self).__init__(dtype,
                                        is_continues,
                                        is_reparameterized=False,
                                        # reparameterization trick is not applied for Bernoulli distribution
                                        group_ndims=group_ndims,
                                        device=device,
                                        **kwargs)

    @property
    def probs(self):
        return self._probs

    @property
    def logits(self):
        return torch.log(self._probs / (torch.ones(self._probs.shape) - self._probs))

    def _batch_shape(self):
        return self.probs.shape

    def _sample(self, n_samples: int = 1, **kwargs):
        if n_samples > 1:
            sample_shape = np.concatenate([[n_samples], self.batch_shape], axis=0).astype(np.int32).tolist()
            _probs = self._probs * torch.ones(tuple(sample_shape)).to(self.device)
        else:
            _probs = self._probs  # * torch.ones(self.batch_shape)

        # _probs *= torch.tensor(_probs <= 1, dtype=self._dtype) #! Values larger than 1 are set to 0
        _sample = torch.bernoulli(_probs)
        self.sample_cache = _sample
        return _sample

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        if len(sample.shape) > len(self._probs.shape):
            sample_shape = np.concatenate([[sample.shape[0]], self.batch_shape], axis=0).astype(np.int32).tolist()
            _probs = self._probs * torch.ones(tuple(sample_shape)).to(self.device)
        else:
            _probs = self._probs  # * torch.ones(self.batch_shape)

        log_prob = sample * torch.log(_probs + 1e-8) + (1 - sample) * torch.log(1 - _probs + 1e-8)
        return log_prob
        # ! Check it again
