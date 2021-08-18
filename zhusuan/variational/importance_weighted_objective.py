import torch
import torch.nn as nn

from zhusuan import log_mean_exp
__all__ = [
    'iw_objective',
    'ImportanceWeightedObjective',
]


class ImportanceWeightedObjective(nn.Module):
    def __init__(self, generator, variational, axis=None):
        super().__init__()
        self.generator = generator
        self.variational = variational

        if axis is None:
            raise ValueError(
                "ImportanceWeightedObjective is a multi-sample objective, "
                "the `axis` argument must be specified.")
        self._axis = axis

    def log_joint(self, nodes):
        log_joint_ = None
        for n_name in nodes.keys():
            try:
                log_joint_ += nodes[n_name].log_prob()
            except:
                log_joint_ = nodes[n_name].log_prob()

        return log_joint_

    def forward(self, observed, reduce_mean=True):
        nodes_q = self.variational(observed).nodes

        _v_inputs = {k:v.tensor for k,v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        nodes_p = self.generator(_observed).nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)
        lower_bound = logpxz - logqz

        if self._axis is not None:
            lower_bound = log_mean_exp(lower_bound, self._axis)

        if reduce_mean:
            return torch.mean(-lower_bound)
        else:
            return -lower_bound

iw_objective = 'ImportanceWeightedObjective',
