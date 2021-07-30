import torch
import torch.nn as nn


class ELBO(nn.Module):
    """
    Short cut for class :class:`~zhusuan.variational.EvidenceLowerBoundObjective`
    """
    def __init__(self, generator, variational):
        super(ELBO, self).__init__()

        self.generator = generator
        self.variational = variational

    def log_joint(self, nodes):
        """
        The default log joint probability function.
        It works by summing over all the conditional log probabilities of
        stochastic nodes evaluated at their current values (samples or
        observations).

        :return: A Var.
        """
        log_joint_ = None
        for n_name in nodes.keys():
            try:
                log_joint_ += nodes[n_name].log_prob()
            except:
                log_joint_ = nodes[n_name].log_prob() #TODO: figure it out
        return log_joint_

    def forward(self, observed, reduce_mean=True):
        nodes_q = self.variational(observed).nodes

        _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        nodes_p = self.generator(_observed).nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)
        # sgvb
        if len(logqz.shape) > 0 and reduce_mean:
            elbo = torch.mean(logpxz - logqz)
        else:
            elbo = logpxz - logqz
        return -elbo

class EvidenceLowerBoundObjective(ELBO):
    """
    The class that represents the evidence lower bound (ELBO) objective for
    variational inference. It can be constructed like a Jittor's `Module` by passing 2 :class:`~zhusuan.framework.bn.BayesianNet` instances. For example, the generator network and the variational inference network in VAE. The model can calculate the ELBO's value with observations passed.

    .. seealso::
        For more details and examples, please refer to
        :doc:`/tutorials/vae` and :doc:`/tutorials/bnn`

    :param generator: A :class:`~zhusuan.framework.bn.BayesianNet` instance that typically defines the learning process.
    :param variational: A :class:`~zhusuan.framework.bn.BayesianNet` instance that defines the variational family.
    """
    def __init__(self, generator, variational):
        super().__init__(generator, variational)