import torch
import torch.nn as nn
from torch import Tensor
from zhusuan.framework import BayesianNet
from zhusuan.distributions import Normal

__all__ = [
    "HMC"
]


class HMC(nn.Module):
    """
    HMC
    """

    def __init__(self, step_size=0.25, n_leapfrogs=25, iters=1):
        super(HMC, self).__init__()
        self.t = 0
        self.step_size = step_size
        self.n_leapfrogs = n_leapfrogs
        self.iters = iters

    def forward(self, bn: BayesianNet, observed: dict, initial_position):

        if initial_position:
            observed_ = {**initial_position, **observed}
        else:
            observed_ = observed
        bn.forward(observed_)

        # interest values q
        q0: dict[str, Tensor] = {}
        for k, v in bn.nodes.items():
            if k not in observed.keys():
                q0[k] = v.sample()

        normals = {}
        for k, v in q0.items():
            normals[k] = Normal(mean=torch.zeros(v.shape), std=torch.ones(v.shape))

        for e in range(self.iters):
            q1: dict[str, Tensor] = {}
            p0: dict[str, Tensor] = {}
            p1: dict[str, Tensor] = {}
            for k, v in q0.items():
                q1[k] = v.clone()
            for k, v in normals.items():
                p0[k] = v.sample()
            for k, v in p0.items():
                p1[k] = v.clone()

            # leapfrog process
            for s in range(self.n_leapfrogs):
                observed_ = {**q1, **observed}
                bn.forward(observed_)
                log_joint_ = bn.log_joint()
                q_v = [v for k, v in q1.items()]
                q_k = [k for k, v in q1.items()]
                q_grad = torch.autograd.grad(log_joint_, q_v)

                for i in range(len(q_grad)):
                    key = q_k[i]
                    p1[key] = p1[key] + self.step_size * q_grad[i] / 2.
                    q1[key] = q1[key] + self.step_size * p1[key]
                    p1[key] = p1[key].detach()
                    q1[key] = q1[key].detach()
                    p1[key].requires_grad = True
                    q1[key].requires_grad = True

                observed_ = {**q1, **observed}
                q_v = [v for k, v in q1.items()]
                q_k = [k for k, v in q1.items()]
                bn.forward(observed_)
                log_joint_ = bn.log_joint()
                q_grad = torch.autograd.grad(log_joint_, q_v)

                for i in range(len(q_grad)):
                    key = q_k[i]
                    p1[key] = p1[key] + self.step_size * q_grad[i] / 2.
                    p1[key] = p1[key].detach()
                    p1[key].requires_grad = True

                # reverse p1
                for k, v in p1.items():
                    p1[k] = -p1[k]

                # M-H step
                observed_ = {**q0, **observed}
                bn.forward(observed_)
                log_prob_q0 = bn.log_joint()
                log_prob_p0 = None
                for k, v in p0.items():
                    len_q = len(log_prob_q0.shape)
                    len_p = len(p0[k].shape)
                    assert (len_p >= len_q)


    def sample(self, bn, observed, initial_position=None):
        """
            sample
        """
        return self.forward(bn, observed, initial_position)
