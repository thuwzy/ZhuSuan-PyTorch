import torch
from zhusuan.distributions import Distribution

class Logistic(Distribution):
    """
    The class of univariate Logistic distribution, using the reparametrization trick from (Kingma, 2013).
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param loc: A 'float' Var. The location term acting on standard Logistic distribution.
    :param scale: A 'float' Var. The scale term acting on standard Logistic distribution.
    """
    def __init__(self,
                dtype=torch.float32,
                is_continues=True,
                group_ndims=0,
                device=torch.device('cpu'),
                **kwargs):
        super(Logistic, self).__init__(dtype,
                                       is_continues,
                                       is_reparameterized = True, 
                                       group_ndims=group_ndims,
                                       device=device,
                                       **kwargs)
        self._loc = torch.as_tensor(kwargs['loc'], dtype = self._dtype).to(device) if type(kwargs['loc']) in [int, float] else kwargs['loc'].to(device)
        self._scale = torch.as_tensor(kwargs['scale'], dtype = self._dtype).to(device) if type(kwargs['scale']) in [int, float] else kwargs['scale'].to(device)
    
    def _batch_shape(self):
        return self._loc.shape
    
    def _sample(self, n_samples=1, **kwargs):
        if n_samples > 1:
            _shape = self._loc.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._loc.shape)
            _loc = self._loc.repeat([n_samples, *_len * [1]])
            _scale = self._scale.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._loc.shape
            _loc = self._loc
            _scale = self._scale

        uniform = torch.nn.init.uniform_(torch.empty(_shape, dtype = self._dtype), 0., 1.) #!check efficiency
        epsilon = torch.log(uniform) - torch.log(1 - uniform)
        _sample = _loc + _scale * epsilon
        self.sample_cache = _sample
        return _sample
    
    def _log_prob(self, sample=None, **kwargs):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._loc.shape):
            n_samples = sample.shape[0]
            _len = len(self._loc.shape)
            _loc = self._loc.repeat([n_samples, *_len * [1]])
            _scale = self._scale.repeat([n_samples, *_len * [1]])
        else:
            _loc = self._loc
            _scale = self._scale

        z = (sample - _loc) / _scale
        return -z - 2. * torch.nn.Softplus()(-z) - torch.log(_scale)


        