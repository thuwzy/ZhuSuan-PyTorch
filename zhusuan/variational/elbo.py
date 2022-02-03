import torch
import torch.nn as nn


class ELBO(nn.Module):
    """
    Short cut for class :class:`~zhusuan.variational.EvidenceLowerBoundObjective`
    """
    def __init__(self, generator, variational, estimator='sgvb'):
        super(ELBO, self).__init__()

        self.generator = generator
        self.variational = variational

        supported_estimator = ['sgvb', 'reinforce']

        if estimator not in supported_estimator:
            raise NotImplementedError()
        self.estimator = estimator
        if estimator == 'reinforce':
            mm = torch.zeros(size=[1], dtype=torch.float32)
            ls = torch.zeros(size=[1], dtype=torch.int32)
            self.register_buffer('moving_mean', mm)
            self.register_buffer('local_step', ls)
            self.moving_mean.requires_grad = False

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

    def forward(self, observed, reduce_mean=True, **kwargs):
        self.variational(observed)
        nodes_q = self.variational.nodes

        _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        self.generator(_observed)
        nodes_p = self.generator.nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)

        if self.estimator == "sgvb":
            return self.sgvb(logpxz, logqz, reduce_mean)
        elif self.estimator == "reinforce":
            return self.reinforce(logpxz, logqz, reduce_mean, **kwargs)

    def sgvb(self, logpxz, logqz, reduce_mean=True):
        # sgvb
        if len(logqz.shape) > 0 and reduce_mean:
            elbo = torch.mean(logpxz - logqz)
        else:
            elbo = logpxz - logqz
        return -elbo

    def reinforce(self, logpxz, logqz, reduce_mean=True, baseline=None, variance_reduction=True, decay=0.8):
        decay_tensor = torch.ones(size=[1], dtype=torch.float32) * decay
        l_signal = logpxz - logqz
        l_signal.require_grads = False # check here
        baseline_cost = None
        if variance_reduction:
            if baseline is not None:
                baseline_cost = 0.5 * torch.square(
                    l_signal.detach() - baseline
                )
                if len(logqz.shape) > 0 and reduce_mean:
                    baseline_cost = torch.mean(baseline_cost)
            l_signal = l_signal - baseline
            # TODO: extend to non-scalar
            if len(logqz.shape) > 0 and reduce_mean:
                bc = torch.mean(l_signal)
            else:
                bc = l_signal
            # Moving average
            self.moving_mean -= (self.moving_mean - bc) * (1.0 - decay)
            self.local_step += 1
            bias_factor = 1 - torch.pow(decay_tensor, self.local_step)
            self.moving_mean /= bias_factor
            l_signal -= self.moving_mean.detach()
        l_signal = l_signal.detach()
        l_signal.require_grads = False # check here
        cost = -logpxz - l_signal * logqz
        if baseline_cost is not None:
            if len(logqz.shape) > 0 and reduce_mean:
                loss = torch.mean(cost + baseline_cost)
            else:
                loss = cost + baseline_cost
            return loss, torch.mean(logpxz - logqz)
        else:
            if len(logqz.shape) > 0 and reduce_mean:
                cost = torch.mean(cost)
            return cost

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
    def __init__(self, generator, variational, estimator='sgvb'):
        super().__init__(generator, variational, estimator)