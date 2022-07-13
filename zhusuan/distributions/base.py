import torch


__all__ = [
    'Distribution',
]

class Distribution(object):
    """
    The :class:`Distribution` class is the base class for various probabilistic
    distributions which support batch inputs, generating batches of samples and
    evaluate probabilities at batches of given values.

    The typical input shape for a :class:`Distribution` is like
    ``batch_shape + input_shape``. where ``input_shape`` represents the shape
    of non-batch input parameter, :attr:`batch_shape` represents how many
    independent inputs are fed into the distribution.

    Samples generated are of shape
    ``([n_samples]+ )batch_shape + value_shape``. The first additional axis
    is omitted only when passed `n_samples` is None (by default), in which
    case one sample is generated. :attr:`value_shape` is the non-batch value
    shape of the distribution. For a univariate distribution, its
    :attr:`value_shape` is [].

    There are cases where a batch of random variables are grouped into a
    single event so that their probabilities should be computed together. This
    is achieved by setting `group_ndims` argument, which defaults to 0.
    The last `group_ndims` number of axes in :attr:`batch_shape` are
    grouped into a single event. For example,
    ``Normal(..., group_ndims=1)`` will set the last axis of its
    :attr:`batch_shape` to a single event, i.e., a multivariate Normal with
    identity covariance matrix.

    When evaluating probabilities at given values, the given Tensor should be
    broadcastable to shape ``(... + )batch_shape + value_shape``. The returned
    Tensor has shape ``(... + )batch_shape[:-group_ndims]``.

    .. seealso::

        For more details and examples, please refer to
        :doc:`/tutorials/concepts`.

    For both, the parameter `dtype` represents type of samples. For discrete,
    can be set by user. For continuous, automatically determined from parameter
    types.

    `dtype` must be among `torch.int16`, `torch.int32`, `torch.int64`,
    `torch.float16`, `torch.float32` and `torch.float64`.

    When two or more parameters are tensors and they have different type,
    `TypeError` will be raised.

    :param dtype: The value type of samples from the distribution.
    :param is_continuous: Whether the distribution is continuous.
    :param is_reparameterized: A bool. Whether the gradients of samples can
        and are allowed to propagate back into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param use_path_derivative: A bool. Whether when taking the gradients
        of the log-probability to propagate them through the parameters
        of the distribution (False meaning you do propagate them). This
        is based on the paper "Sticking the Landing: Simple,
        Lower-Variance Gradient Estimators for Variational Inference"
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in :attr:`batch_shape` (counted from the end) that are
        grouped into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See above for more detailed explanation.
    """    
    def __init__(self,
                 dtype,
                 is_continuous,
                 is_reparameterized,
                 use_path_derivative=False,
                 group_ndims=0,
                 device=torch.device('cpu'),
                 **kwargs):

        self._dtype = dtype
        self._is_continuous = is_continuous
        self._is_reparameterized = is_reparameterized
        self._use_path_derivative = use_path_derivative
        self._device = device
    
        if isinstance(group_ndims, int):
            if group_ndims < 0:
                raise ValueError("group_ndims must be non-negative.")
            self._group_ndims = group_ndims
        else:
            #TODO
            pass

    @property
    def dtype(self):
        """The sample type of the distribution."""
        return self._dtype

    @property
    def device(self):
        """
        The device this distribution lies at.
        
        :return: torch.device
        """     
        return self._device

    @property
    def is_reparameterized(self):
        """
        Whether the gradients of samples can and are allowed to propagate back
        into inputs, using the reparametrization trick from (Kingma, 2013).
        """
        return self._is_reparameterized

    @property
    def batch_shape(self):
        """
        The shape showing how many independent inputs (which we call batches)
        are fed into the distribution. For batch inputs, the shape of a
        generated sample is ``batch_shape + value_shape``.
        """
        #TODO
        return self._batch_shape()

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        raise NotImplementedError()

    def sample(self, n_samples=None):
        """
        sample(n_samples=None)
        
        Return samples from the distribution. When `n_samples` is None (by
        default), one sample of shape ``batch_shape + value_shape`` is
        generated. For a scalar `n_samples`, the returned Var has a new
        sample dimension with size `n_samples` inserted at ``axis=0``, i.e.,
        the shape of samples is ``[n_samples] + batch_shape + value_shape``.
        
        :param n_samples: A 0-D `int32` Tensor or None. How many independent
            samples to draw from the distribution.
        :return: A Var of samples.
        """
        if n_samples is None:
            samples = self._sample(n_samples=1)
            return samples
        elif isinstance(n_samples, int):
            return self._sample(n_samples)
        else:
            #TODO
            pass

    def _sample(self, n_samples):
        """
        Private method for subclasses to rewrite the :meth:`sample` method.
        """
        raise NotImplementedError()

    def log_prob(self, given):
        """
        log_prob(given)
        
        Compute log probability density (mass) function at `given` value.
        
        :param given: A Var. The value at which to evaluate log probability
            density (mass) function. Must be able to broadcast to have a shape
            of ``(... + )batch_shape + value_shape``.
        :return: A Var of shape ``(... + )batch_shape[:-group_ndims]``.
        """
        log_p = self._log_prob(given)
        if self._group_ndims > 0:
            return torch.sum(log_p, [i for i in range(-self._group_ndims, 0)])
        else:
            return log_p

    def _log_prob(self, given):
        """
        Private method for subclasses to rewrite the :meth:`log_prob` method.
        """
        raise NotImplementedError()