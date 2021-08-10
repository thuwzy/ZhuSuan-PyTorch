import torch
import torch.nn as nn

from zhusuan.framework.stochastic_tensor import StochasticTensor
from zhusuan.distributions import *
from zhusuan.flow import *

class BayesianNet(nn.Module):
    def __init__(self, observed=None):
        """
        The :class:`BayesianNet` class provides a convenient way to construct
        Bayesian networks, i.e., directed graphical models.

        To start, we create a :class:`BayesianNet` instance::
            
            class Net(BayesianNet):
                def __init__(self):
                    # Initialize...
                def execute(self, observed):
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

    def sn(self, *args, **kwargs):
        """
        Short cut for method :meth:`~zhusuan.framework.bn.BayesianNet.stochastic_node`
        """
        return self.stochastic_node(*args, **kwargs)

    def stochastic_node(self, distribution, name, **kwargs):
        """
        Add a stochastic node in this :class:`BayesianNet` that follows the distribution assigned by the `name` parameter.

        :param distribution: The distribution which the node follows.
        :param name: The unique name of the node.
        :param **kwargs: Parameters of the distribution which the node builds with.
        :return: A instance(sample) of the node.
        """
        _dist = globals()[distribution](**kwargs) #TODO: `globals()` is unsafe
        self._nodes[name] = StochasticTensor(self, name, _dist, **kwargs)
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
        #TODO: Why summing over?
        if use_cache:
            if not hasattr(self, '_log_joint_cache'):
                self._log_joint_cache = self._log_joint()
        else:
            self._log_joint_cache = self._log_joint()
        return self._log_joint_cache