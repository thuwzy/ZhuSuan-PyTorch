import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from zhusuan.framework.stochastic_tensor import StochasticTensor

from zhusuan import log_mean_exp

__all__ = [
    'iw_objective',
    'ImportanceWeightedObjective',
]


def compute_iw_term(x, axis):
    """
    compute the importance weighted gradient estimation using Numerically stable method
    param x: log(w) term
    """
    x_max = torch.max(x, axis, keepdim=True).values
    w: torch.Tensor = (x - x_max).exp()
    w_tilde = (w / w.sum(dim=axis, keepdim=True)).detach()
    return (w_tilde * x).sum(axis)


class ImportanceWeightedObjective(nn.Module):
    def __init__(self, generator, variational, axis=None, estimator='sgvb'):
        super().__init__()
        self.generator = generator
        self.variational = variational

        if axis is None:
            raise ValueError(
                "ImportanceWeightedObjective is a multi-sample objective, "
                "the `axis` argument must be specified.")
        self._axis = axis

        supported_estimator = ['sgvb', 'vimco', "naive"]
        if estimator not in supported_estimator:
            raise NotImplementedError()
        self.estimator = estimator

    def log_joint(self, nodes):
        log_joint_ = None
        for n_name in nodes.keys():
            adder = nodes[n_name].log_prob()
            if log_joint_ is None:
                log_joint_ = torch.zeros(adder.shape)
            log_joint_ += adder

            # warnings.warn(f"exception getted in log_joint method, using current node {n_name}")
            # log_joint_ = nodes[n_name].log_prob()

        return log_joint_

    def forward(self, observed, reduce_mean=True):
        # feed forward observation to the variational(comments are based on iwae case)
        self.variational(observed)
        nodes_q: dict = self.variational.nodes
        _v_inputs = {}
        for k, v in nodes_q.items():
            _v_inputs[k] = v.tensor
            if self.estimator == "vimco" \
                and isinstance(v, StochasticTensor)\
                    and v.dist.is_reparameterized:
                raise ValueError("with vimco estimator, the is_reparameterized must be false")

        _observed = {**_v_inputs, **observed}
        nodes_p = self.generator(_observed).nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)

        if self.estimator == 'sgvb':
            return self.sgvb(logpxz, logqz, reduce_mean)
        elif self.estimator == 'naive':
            return self.naive(logpxz, logqz, reduce_mean)
        else:
            return self.vimco(logpxz, logqz, reduce_mean)

    def sgvb(self, logpxz, logqz, reduce_mean=True):
        log_w = logpxz - logqz
        if self._axis is not None:
            lower_bound = compute_iw_term(log_w, self._axis)
        else:
            lower_bound = log_w

        if reduce_mean:
            return torch.mean(-lower_bound)
        else:
            return -lower_bound

    def naive(self, logpxz, logqz, reduce_mean=True):
        log_w = logpxz - logqz

        l_signal = log_mean_exp(log_w, self._axis, keepdims=True)
        fake_term = torch.sum(logqz * l_signal.detach(), self._axis)

        if self._axis is not None:
            iw_term = compute_iw_term(log_w, self._axis)
            log_w = iw_term + fake_term

        if reduce_mean:
            return torch.mean(-log_w)
        else:
            return - log_w

    def vimco(self, logpxz, logqz, reduce_mean=True):
        log_w = logpxz - logqz
        # check size along the sample axis
        err_msg = "VIMCO is a multi-sample gradient estimator, size along " \
                  "`axis` in the objective should be larger than 1."
        try:
            _shape = log_w.shape
            _ = _shape[self._axis:self._axis + 1]
            if _shape[self._axis] < 2:
                raise ValueError(err_msg)
        except:
            raise ValueError(err_msg)

        l_signal = log_w
        # compute the mean of sample dim, the inplace position value is dropped,
        # in iwae case: [sample dim, batch size]
        mean_expect_signal = (torch.sum(l_signal, dim=self._axis, keepdim=True) - l_signal) \
                             / torch.as_tensor(l_signal.shape[self._axis] - 1, dtype=l_signal.dtype)
        x, sub_x = l_signal, mean_expect_signal
        int_dim = x.dim()
        n_dim = torch.as_tensor(int_dim, dtype=torch.int32)
        # mark the sample axis as bool dim mask
        axis_dim_mask = torch.as_tensor(F.one_hot(torch.tensor(self._axis), num_classes=n_dim), dtype=torch.bool)
        original_mask = torch.as_tensor(F.one_hot(torch.tensor(x.dim() - 1), num_classes=n_dim), dtype=torch.bool)
        axis_dim = torch.ones(int_dim, dtype=torch.int32) * self._axis
        originals = torch.ones(int_dim, dtype=torch.int32) * (int_dim - 1)
        perm = torch.where(original_mask, axis_dim, torch.arange(int_dim, dtype=torch.int32))
        perm = torch.where(axis_dim_mask, originals, perm)
        multiples = torch.concat([torch.ones([int_dim], dtype=torch.int32), torch.tensor([x.shape[self._axis]])], 0)
        if len(perm) > 1:
            # exchange the sample dim and last dim
            x = torch.transpose(x, *list(perm))
            sub_x = torch.transpose(sub_x, *list(perm))
        x_ex = torch.tile(torch.unsqueeze(x, int_dim), tuple(multiples))
        x_ex = x_ex - torch.diag_embed(x) + torch.diag_embed(sub_x)
        control_variate = torch.permute(log_mean_exp(x_ex, int_dim - 1), list(perm))
        l_signal = log_mean_exp(l_signal, self._axis, keepdims=True) - control_variate
        fake_term = torch.sum(logqz * l_signal.detach(), self._axis)
        iw_term = compute_iw_term(log_w, self._axis)
        cost = -fake_term - iw_term
        return cost.mean()


iw_objective = 'ImportanceWeightedObjective',
