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


class Coupling(RevNet):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)

    def _forward(self, x, **kwargs):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W)), None


    def _inverse(self, x, **kwargs):
        [B, W] = list(x.size())
        x = x.reshape((B, W // 2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        on = on - shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W)), None