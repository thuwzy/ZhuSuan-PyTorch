from __future__ import absolute_import
from __future__ import division
import torch
import torch.nn as nn

from zhusuan.invertible.base import RevNet

__all__ = [
    "Scaling"
]


class Scaling(RevNet):
    def __init__(self, dim):
        super(Scaling, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros([1, dim]), requires_grad=True)

    def _forward(self, x, **kwargs):
        log_det_J = torch.sum(self.log_scale)
        x *= self.log_scale.exp()
        return x, log_det_J

    def _inverse(self, y, **kwargs):
        log_det_J = torch.sum(self.log_scale)
        y *= torch.exp(-self.log_scale)
        return y, log_det_J
