import torch
import torch.nn as nn

from zhusuan import log_mean_exp
__all__ = [
    'iw_objective',
    'ImportanceWeightedObjective',
]


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

        supported_estimator = ['sgvb', 'vimco']
        if estimator not in supported_estimator:
            raise NotImplementedError()
        self.estimator = estimator


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

        if self.estimator == 'sgvb':
            return self.sgvb(logpxz, logqz, reduce_mean)
        else:
            return self.vimco(logpxz, logqz, reduce_mean)

    def sgvb(self, logpxz, logqz, reduce_mean=True):
        lower_bound = logpxz - logqz

        if self._axis is not None:
            lower_bound = log_mean_exp(lower_bound, self._axis)

        if reduce_mean:
            return torch.mean(-lower_bound)
        else:
            return -lower_bound

    def vimco(self, logpxz, logqz, reduce_mean=True):
        log_w = logpxz - logqz
        l_signal = log_w

        # check size along the sample axis
        err_msg = "VIMCO is a multi-sample gradient estimator, size along " \
                  "`axis` in the objective should be larger than 1."
        try:
            _shape = l_signal.shape
            _ = _shape[self._axis:self._axis + 1]
            if _shape[self._axis] < 2:
                raise ValueError(err_msg)
        except:
            raise ValueError(err_msg)

        # compute variance reduction term
        mean_expect_signal = (torch.sum(l_signal, dim = self._axis, keepdim=True) - l_signal) \
                            / torch.as_tensor(l_signal.shape[self._axis] - 1, dtype=l_signal.dtype)
        x, sub_x = l_signal, mean_expect_signal
        n_dim = torch.as_tensor(x.dim(), dtype=torch.int32)
        n_dim = torch.unsqueeze(n_dim, -1)

        axis_dim_mask = torch.as_tensor(torch.nn.functional.one_hot(torch.LongTensor([self._axis]), num_classes = n_dim.numpy()[0]), dtype=torch.bool)
        original_mask = torch.as_tensor(torch.nn.functional.one_hot(torch.LongTensor([n_dim - 1]), num_classes = n_dim.numpy()[0]), dtype=torch.bool)
        axis_dim = torch.ones(n_dim) * self._axis
        originals = torch.ones(n_dim) * (n_dim - 1)
        perm = torch.where(torch.squeeze(original_mask, dim=-1), axis_dim, torch.arange(n_dim.numpy()[0], dtype=torch.float32))
        perm = torch.where(torch.squeeze(axis_dim_mask, dim=-1), originals, perm)
        multiples = torch.cat([torch.ones(n_dim), torch.tensor([x.shape[self._axis]])], 0)
        multiples = tuple([int(i) for i in multiples.numpy()])

        perm = tuple([int(i) for i in perm.numpy()])
        x = torch.permute(x, perm)
        sub_x = torch.permute(sub_x, perm)
        x_ex = torch.tile(torch.unsqueeze(x, n_dim.numpy()[0]), multiples)
        x_ex = x_ex - torch.diag(x) + torch.diag(sub_x)
        control_variate = torch.permute(log_mean_exp(x_ex, n_dim.numpy()[0] - 1), perm)

        # variance reduced objective
        l_signal = log_mean_exp(l_signal, self._axis, keepdims=True) - control_variate
        fake_term = torch.sum(logqz * l_signal.detach(), self._axis)
        cost = -fake_term - log_mean_exp(log_w, self._axis)

        return cost


iw_objective = 'ImportanceWeightedObjective',
