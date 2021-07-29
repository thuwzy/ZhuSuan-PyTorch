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

    The value type of `prob` and `log_prob` will be `param_dtype` which is
    deduced from the parameter(s) when initializating. And `dtype` must be
    among `int16`, `int32`, `int64`, `float16`, `float32` and `float64`.

    When two or more parameters are tensors and they have different type,
    `TypeError` will be raised.

    :param dtype: The value type of samples from the distribution.
    :param param_dtype: The parameter(s) type of the distribution.
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
                 param_dtype,
                 is_continuous,
                 is_reparameterized,
                 use_path_derivative=False,
                 group_ndims=0,
                 **kwargs):
                 
        self._dtype = dtype
        self._param_dtype = param_dtype
        self._is_continuous = is_continuous
        self._is_reparameterized = is_reparameterized
        self._use_path_derivative = use_path_derivative