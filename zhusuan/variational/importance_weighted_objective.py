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
    param axis: the samples axis the importance term that will be computed
    """
    x_max = torch.max(x, axis, keepdim=True).values
    w: torch.Tensor = (x - x_max).exp()
    w_tilde = (w / w.sum(dim=axis, keepdim=True)).detach()
    return (w_tilde * x).sum(axis)


class ImportanceWeightedObjective(nn.Module):
    """
    The class that represents the importance weighted objective for
    variational inference (Burda, 2015)

    As a variational objective, :class:`ImportanceWeightedObjective` provides two
    gradient estimators for the variational (proposal) parameters:

    * :meth:`sgvb`: The Stochastic Gradient Variational Bayes (SGVB) estimator,
      also known as "the reparameterization trick", or "path derivative
      estimator".
    * :meth:`vimco`: The multi-sample score function estimator with variance
      reduction, also known as "VIMCO".

    :param generator: generator part of importance weighted objective
    :param variational: variational part of importance weighted objective
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in the objective.
    :param estimator: the estimator, a str in either 'sgvb' or 'vimco'

    """

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
                    and isinstance(v, StochasticTensor) \
                    and v.dist.is_reparameterized:
                raise ValueError("with vimco estimator, the is_reparameterized must be false")

        _observed = {**_v_inputs, **observed}
        nodes_p = self.generator(_observed).nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)

        if self.estimator == 'sgvb':
            return self.sgvb(logpxz, logqz, reduce_mean)
        else:  # self.estimator == 'vimco'
            return self.vimco(logpxz, logqz, reduce_mean)

    def sgvb(self, logpxz, logqz, reduce_mean=True):
        """
           Implements the stochastic gradient variational bayes (SGVB) gradient
           estimator for the objective, also known as "reparameterization trick"
           or "path derivative estimator". It was first used for importance
           weighted objectives in (Burda, 2015), where it's named "IWAE".

           It only works for latent `StochasticTensor` s that can be
           reparameterized (Kingma, 2013). For example,
           :class:`~zhusuan.distribution.Normal`
           and :class:`~zhusuan.framework.stochastic.Concrete`.

           .. note::

               To use the :meth:`sgvb` estimator, the ``is_reparameterized``
               property of each latent `StochasticTensor` must be True (which is
               the default setting when they are constructed).

           :return: A Tensor. The surrogate cost for optimizers to
               minimize.
        """
        log_w = logpxz - logqz
        if self._axis is not None:
            lower_bound = compute_iw_term(log_w, self._axis)
        else:
            lower_bound = log_w

        if reduce_mean:
            return torch.mean(-lower_bound)
        else:
            return -lower_bound

    def vimco(self, logpxz, logqz, reduce_mean=True):
        """
        Implements the multi-sample score function gradient estimator for
        the objective, also known as "VIMCO", which is named
        by authors of the original paper (Minh, 2016).

        It works for all kinds of latent `StochasticTensor` s.

        .. note::

            To use the :meth:`vimco` estimator, the ``is_reparameterized``
            property of each reparameterizable latent `StochasticTensor` must
            be set False.

        :return: A Tensor. The surrogate cost for optimizers to
            minimize.
        """

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
        multiples = torch.cat([torch.ones([int_dim], dtype=torch.int32), torch.tensor([x.shape[self._axis]])], 0)
        if len(perm) > 1:
            # exchange the sample dim and last dim
            x = torch.transpose(x, *list(perm))
            sub_x = torch.transpose(sub_x, *list(perm))
        x_ex = torch.unsqueeze(x, int_dim).repeat(tuple(multiples))
        x_ex = x_ex - torch.diag_embed(x) + torch.diag_embed(sub_x)
        control_variate = torch.permute(log_mean_exp(x_ex, int_dim - 1), list(perm))
        l_signal = log_mean_exp(l_signal, self._axis, keepdims=True) - control_variate
        fake_term = torch.sum(logqz * l_signal.detach(), self._axis)
        iw_term = compute_iw_term(log_w, self._axis)
        cost = -fake_term - iw_term
        return cost.mean()


iw_objective = 'ImportanceWeightedObjective',
