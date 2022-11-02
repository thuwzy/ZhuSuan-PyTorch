import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from zhusuan.invertible import RevNet


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)
        self.register_buffer("mask", mask)
        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) /
                                            math.sqrt(cond_label_size))

    def forward(self, x, cond_y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if cond_y is not None:
            out = out + F.linear(cond_y, self.cond_weight)
        return out


class MADE(RevNet):
    # maily from normalizing_flows/maf.py
    @staticmethod
    def create_mask(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
        """
        Mask generator for MADE & MAF (see MADE paper sec 4:https://arxiv.org/abs/1502.03509)
        Args:
            input_size:
            hidden_size:
            n_hidden:
            input_order:
            input_degrees:

        Returns: List of masks

        """
        degrees = []
        if input_order == "sequential":
            degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                degrees += [torch.arange(hidden_size) % (input_size - 1)]
            degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [
                input_degrees % input_size - 1]
        elif input_order == 'random':
            degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
                degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [
                input_degrees - 1]
        else:
            raise NotImplementedError("input_order must be in \'sequential\' or \'random\'")

        # construct masks
        masks = []
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]
        return masks, degrees[0]

    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None,
                 input_order="sequential", input_degrees=None, activation="relu"):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super(MADE, self).__init__()
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create mask
        masks, self.input_degrees = self.create_mask(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]

        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    def _forward(self, x, cond_y=None, **kwargs):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, cond_y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def _inverse(self, u, cond_y=None, **kwargs):
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            y = self.net_input(x, cond_y)
            m, loga = self.net(y).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        log_det = loga
        return x, log_det