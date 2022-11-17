import torch
from torch._C import device
import torch.nn as nn

from zhusuan.framework.stochastic_tensor import StochasticTensor
from zhusuan.distributions import *

name_mapping = {
    "Normal": Normal,
    "Bernoulli": Bernoulli,
    "Beta": Beta,
    "Exponential": Exponential,
    "Gamma": Gamma,
    "Laplace": Laplace,
    "Logistic": Logistic,
    "Poisson": Poisson,
    "StudentT": StudentT,
    "Uniform": Uniform
}


class BayesianNet(nn.Module):
    def __init__(self, observed=None, device=torch.device('cpu')):
        """
        The :class:`BayesianNet` class provides a convenient way to construct
        Bayesian networks, i.e., directed graphical models.

        To start, we create a :class:`BayesianNet` instance::
            
            class Net(BayesianNet):
                def __init__(self):
                    # Initialize...
                def forward(self, observed):
                    # Forward propagation...
        
        A :class:`BayesianNet` keeps two kinds of nodes
        * deterministic nodes: they are just Vars, usually the outputs of Jittor operations.
        * stochastic nodes: they are random variables in graphical models, and can be constructed inside the ``BayesianNet`` class like
        
        ::
        
            self.stochastic_node('Normal', name='w', mean=0., std=1.)
        
        To observe any stochastic nodes in the network, pass a dictionary mapping
        of ``(name, Tensor)`` pairs when call the model which derives :class:`BayesianNet`.
        This will assign observed values to corresponding
        :class:`StochasticTensor` s. For example::
            
            net = Net()
            net({'w': w_obs})
        
        will set ``w`` to be observed.
        
        .. note::
            The observation passed must have the same type and shape as the
            :class:`StochasticTensor`.
        
        .. seealso::
            For more details and examples, please refer to
            :doc:`/tutorials/concepts`.
        
        :param observed: A dictionary of (string, Tensor) pairs, which maps from
            names of stochastic nodes to their observed values.
        """
        super(BayesianNet, self).__init__()
        self._nodes = {}
        self._cache = {}
        self._observed = observed if observed else {}

        self._device = device

    @property
    def nodes(self):
        """
        The dictionary of all named stochastic nodes in this :class:`BayesianNet`.
        
        :return: A dict.
        """
        return self._nodes

    @property
    def cache(self):
        """
        The dictionary of all named deterministic nodes in this :class:`BayesianNet`.

        :return: A dict.
        """
        return self._cache

    @property
    def device(self):
        """
        The device this module lies at.
        
        :return: torch.device
        """
        try:
            return next(self.parameters()).device
        except:
            return self._device

    def to(self, device):
        self._device = device
        return super().to(device)

    @property
    def observed(self):
        """
        The dictionary of all observed nodes in this :class:`BayesianNet`.

        :return: A dict.
        """
        return self._observed

    def observe(self, observed):
        """
        Assign the nodes and values to be observed in this :class:`BayesianNet`.

        :param observed: A dictionary of (string, Tensor) pairs, which maps from
            names of stochastic nodes to their observed values.
        """
        self._observed = {}
        for k, v in observed.items():
            self._observed[k] = v
        return self

    def sn(self, dist, name, n_samples=None, **kwargs):
        """
        Short cut for method :meth:`~zhusuan.framework.bn.BayesianNet.stochastic_node`
        """
        return self.stochastic_node(dist, name, n_samples, **kwargs)

    def snode(self, *args, **kwargs):
        """
        Short cut for method :meth:`~zhusuan.framework.bn.BayesianNet.stochastic_node`
        """
        return self.stochastic_node(*args, **kwargs)

    def stochastic_node(self, distribution, name, n_samples=None, **kwargs):
        """
        Add a stochastic node in this :class:`BayesianNet` that follows the distribution assigned by the
        ``name`` parameter.

        :param distribution: The distribution which the node follows.
        :param name: The unique name of the node.
        :param n_samples: number of samples per sample process
        :param kwargs: Parameters of the distribution which the node builds with.
        :return: A instance(sample) of the node.
        """
        if isinstance(distribution, str):
            _dist = name_mapping[distribution](device=self.device, **kwargs)
            self._nodes[name] = StochasticTensor(self, name, _dist, n_samples=n_samples, **kwargs)
        elif isinstance(distribution, Distribution):
            distribution._device = self.device
            self._nodes[name] = StochasticTensor(self, name, distribution, n_samples=n_samples, **kwargs)
        else:
            raise ValueError('distribution must be name of sub class of Distribution or an instance of Distribution')
        return self._nodes[name].tensor

    def _log_joint(self):
        _ret = 0
        for k, v in self._nodes.items():
            if isinstance(v, StochasticTensor):
                try:
                    _ret = _ret + v.log_prob()
                except:
                    _ret = v.log_prob()
        return _ret

    def log_joint(self, use_cache=False):
        """
        The default log joint probability of this :class:`BayesianNet`.
        It works by summing over all the conditional log probabilities of
        stochastic nodes evaluated at their current values (samples or
        observations).

        :return: A Var.
        """
        # TODO: Why summing over?
        if use_cache:
            if not hasattr(self, '_log_joint_cache'):
                self._log_joint_cache = self._log_joint()
        else:
            self._log_joint_cache = self._log_joint()
        return self._log_joint_cache

    # aliases of specific distribution
    def normal(self,
               name,
               mean=0.,
               std=None,
               logstd=None,
               dtype=None,
               is_continuous=True,
               is_reparameterized=True,
               group_ndims=0,
               n_samples=None,
               **kwargs):
        if isinstance(name, str):
            distribution = Normal(
                mean=mean,
                std=std,
                logstd=logstd,
                dtype=dtype,
                is_continuous=is_continuous,
                is_reparameterized=is_reparameterized,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def bernoulli(self,
                  name,
                  logits=None,
                  probs=None,
                  dtype=None,
                  is_continuous=False,
                  group_ndims=0,
                  n_samples=None,
                  **kwargs):
        if isinstance(name, str):
            distribution = Bernoulli(
                logits=logits,
                probs=probs,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def beta(self, name,
             alpha,
             beta,
             dtype=None,
             is_continuous=True,
             group_ndims=0,
             n_samples=None,
             **kwargs):
        if isinstance(name, str):
            distribution = Beta(
                alpha=alpha,
                beta=beta,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def exponential(self, name,
                    rate,
                    dtype=None,
                    is_continuous=True,
                    group_ndims=0,
                    n_samples=None,
                    **kwargs):
        if isinstance(name, str):
            distribution = Exponential(
                rate=rate,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def gamma(self, name,
              alpha,
              beta,
              dtype=None,
              is_continuous=True,
              group_ndims=0,
              n_samples=None,
              **kwargs):
        if isinstance(name, str):
            distribution = Gamma(
                alpha=alpha,
                beta=beta,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def laplace(self, name,
                loc,
                scale,
                dtype=None,
                is_continuous=True,
                group_ndims=0,
                n_samples=None,
                **kwargs):
        if isinstance(name, str):
            distribution = Laplace(
                loc=loc,
                scale=scale,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def logistic(self, name,
                 loc,
                 scale,
                 dtype=None,
                 is_continuous=True,
                 group_ndims=0,
                 n_samples=None,
                 **kwargs):
        if isinstance(name, str):
            distribution = Laplace(
                loc=loc,
                scale=scale,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def poisson(self, name,
                rate,
                dtype=None,
                is_continuous=True,
                group_ndims=0,
                n_samples=None,
                **kwargs):
        if isinstance(name, str):
            distribution = Poisson(
                rate=rate,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def studentT(self, name,
                 df,
                 loc=0.,
                 scale=1.,
                 dtype=None,
                 is_continuous=True,
                 group_ndims=0,
                 n_samples=None,
                 **kwargs):
        if isinstance(name, str):
            distribution = StudentT(
                df=df,
                loc=loc,
                scale=scale,
                dtype=dtype,
                is_continuous=is_continuous,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")

    def uniform(self, name,
                low,
                high,
                dtype=None,
                is_continuous=True,
                is_reparameterized=True,
                group_ndims=0,
                n_samples=None,
                **kwargs):
        if isinstance(name, str):
            distribution = Uniform(
                low=low,
                high=high,
                dtype=dtype,
                is_continuous=is_continuous,
                is_reparameterized=is_reparameterized,
                group_ndims=group_ndims,
                device=self.device,
                **kwargs
            )
            self._nodes[name] = StochasticTensor(self, name, distribution,
                                                 n_samples=n_samples, **kwargs)
            return self._nodes[name].tensor
        else:
            raise ValueError("name of stochastic_node must be str")
