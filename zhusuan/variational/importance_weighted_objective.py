import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            print(n_name)
            adder = nodes[n_name].log_prob()
            print("log prob shape:", adder.shape)
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
        _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
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
        lower_bound = logpxz - logqz

        if self._axis is not None:
            lower_bound = log_mean_exp(lower_bound, self._axis)

        if reduce_mean:
            return torch.mean(-lower_bound)
        else:
            return -lower_bound

    # def naive(self, logpxz, logqz, reduce_mean=True):
    #     lower_bound = logpxz - logqz
    #
    #     l_signal = log_mean_exp(lower_bound.detach(), self._axis, keepdims=True)
    #     fake_term = torch.sum(logqz * l_signal.detach(), self._axis)
    #
    #     if self._axis is not None:
    #         lower_bound = log_mean_exp(lower_bound, self._axis) + fake_term
    #
    #     if reduce_mean:
    #         return torch.mean(-lower_bound)
    #     else:
    #         return -lower_bound

    def vimco(self, logpxz, logqz, reduce_mean=True):
        # TODO
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
        # compute origin L(h^1:k)
        origin_l = log_mean_exp(log_w, dim=self._axis, keepdims=True)
        num_sample = log_w.shape[self._axis]
        w_max = torch.max(log_w, self._axis, True).values
        # compute variance reduction term
        # compute independent sum of f(x,h^i)
        subed = log_w - w_max
        sum_item = (torch.sum(subed.exp(), dim=self._axis, keepdim=True) - subed.exp())
        # compute independent estimate of f(h^j) by geometric mean
        estimate_fhj = (((torch.sum(log_w, dim=self._axis) - log_w) / (num_sample - 1)) - w_max).exp()
        # compute baseline term
        base = torch.log((sum_item + estimate_fhj) / num_sample) + w_max
        l_signal = (origin_l - base).detach()
        print(l_signal)

        # x, sub_x = l_signal, mean_expect_signal
        # int_dim = x.dim()
        # n_dim = torch.as_tensor(x.dim(), dtype=torch.int32)
        # # print(n_dim)
        # axis_dim_mask = torch.as_tensor(F.one_hot(torch.tensor(self._axis), num_classes=n_dim), dtype=torch.bool)
        # # print("axis mask", axis_dim_mask)
        # original_mask = torch.as_tensor(F.one_hot(torch.tensor(x.dim() - 1), num_classes=n_dim), dtype=torch.bool)
        # # print("original mask", original_mask)
        #
        # axis_dim = torch.ones(int_dim, dtype=torch.int32) * self._axis
        # # print("axis_dim", axis_dim)
        # originals = torch.ones(int_dim, dtype=torch.int32) * (int_dim - 1)
        # # print("origins", originals)
        # perm = torch.where(original_mask, axis_dim, torch.arange(int_dim, dtype=torch.int32))
        # perm = torch.where(axis_dim_mask, originals, perm)
        # # print(perm)
        # multiples = torch.concat([torch.ones([int_dim], dtype=torch.int32), torch.tensor([x.shape[self._axis]])], 0)

        # print(multiples)
        # print(x)
        # x = torch.transpose(x, *list(perm))
        # sub_x = torch.transpose(sub_x, *list(perm))
        # x_ex = torch.tile(torch.unsqueeze(x, int_dim), tuple(multiples))
        # # print(x, sub_x)
        # # print("diag",  torch.diag_embed(x)[0], torch.diag_embed(sub_x)[0])
        # x_ex = x_ex - torch.diag_embed(x) + torch.diag_embed(sub_x)
        # # print(x_ex)
        # control_variate = torch.permute(log_mean_exp(x_ex, int_dim - 1), list(perm))
        # print(f"cv {control_variate, control_variate.shape}")

        # perm = torch.where(torch.squeeze(original_mask, dim=-1), axis_dim, torch.arange(n_dim.numpy()[0], dtype=torch.float32))
        # perm = torch.where(torch.squeeze(axis_dim_mask, dim=-1), originals, perm)
        # multiples = torch.cat([torch.ones(n_dim), torch.tensor([x.shape[self._axis]])], 0)
        # multiples = tuple([int(i) for i in multiples.numpy()])
        #
        # # TODO
        # perm = tuple([int(i) for i in perm.numpy()])
        # x = torch.permute(x, perm)
        # sub_x = torch.permute(sub_x, perm)
        # x_ex = torch.tile(torch.unsqueeze(x, n_dim.numpy()[0]), multiples)
        # x_ex = x_ex - torch.diag(x) + torch.diag(sub_x)
        # control_variate = torch.permute(log_mean_exp(x_ex, n_dim.numpy()[0] - 1), perm)

        # l_max = torch.max(l_signal, self._axis, True).values
        # control_variate = torch.log(torch.mean(torch.exp(l_signal - l_max), self._axis, True) +
        #                             (torch.exp(mean_expect_signal - l_max) - torch.exp(l_signal - l_max)) /
        #                             torch.as_tensor(l_signal.shape[self._axis], dtype=l_signal.dtype)) + l_max

        # variance reduced objective

        # l_signal = log_mean_exp(l_signal, self._axis, keepdims=True) - control_variate
        # fake_term = torch.sum(logqz * l_signal.detach(), self._axis)
        # L = fake_term + log_mean_exp(log_w, self._axis)
        L = log_mean_exp(log_w, self._axis)
        # print(fake_term, log_mean_exp(log_w, self._axis))
        return -L.mean()


iw_objective = 'ImportanceWeightedObjective',
