import torch
import math
import torch.nn as nn
from .base import RevNet
from .sequential import RevSequential


def get_coupling_mask(n_dim, n_channel, n_mask, split_type="OddEven"):
    """

    """
    masks = []
    if split_type == "OddEven":
        if n_channel == 1:
            mask = torch.arange(n_dim, dtype=torch.float32) % 2
            for i in range(n_mask):
                masks.append(mask)
                mask = 1. - mask
    elif split_type == "Half":
        pass
    elif split_type == "RandomHalf":
        pass
    else:
        raise NotImplementedError()
    return masks


class AdditiveCoupling(RevNet):
    def __init__(self, in_out_dim=-1, mid_dim=-1, hidden=-1, mask=None, inner_nn=None):
        super(AdditiveCoupling, self).__init__()
        if inner_nn is None:

            stdv = 1. / math.sqrt(mid_dim)
            stdv2 = 1. / math.sqrt(in_out_dim)

            self.nn = []
            self.nn.append(nn.Linear(in_out_dim, mid_dim))
            self.nn.append(nn.ReLU())
            for _ in range(hidden - 1):
                self.nn.append(nn.Linear(mid_dim, mid_dim))
                self.nn.append(nn.ReLU())
            self.nn.append(nn.Linear(mid_dim, in_out_dim))
            self.nn = nn.Sequential(*self.nn)
        else:
            self.nn = inner_nn
        self.mask = mask

    def _forward(self, x, **kwargs):
        x1, x2 = self.mask * x, (1. - self.mask) * x
        shift = self.nn(x1)
        y1, y2 = x1, x2 + shift * (1. - self.mask)
        return y1 + y2, None

    def _inverse(self, y, **kwargs):
        y1, y2 = self.mask * y, (1. - self.mask) * y
        shift = self.nn(y1)
        x1, x2 = y1, y2 - shift * (1. - self.mask)
        return x1 + x2, None
