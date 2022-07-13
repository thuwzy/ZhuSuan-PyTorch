#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

# TODO
def log_mean_exp(x, dim=None, keepdims=False):
    """
    Numerically stable log mean of exps across the `dim`.
    :param x: A Tensor.
    :param dim: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keepdims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.
    :return: A Tensor after the computation of log mean exp along given axes of
        x.
    """
    log_w = x
    log_w_minus_max = log_w - log_w.max(dim, keepdim=True)[0]
    # compute normalized importance weights (no gradient)
    w = log_w_minus_max.exp()
    w_tilde = (w / w.sum(axis=dim, keepdim=True)).detach()
    # compute loss (negative IWAE objective)
    loss = -(w_tilde * log_w).sum(dim)
    print(loss.shape)

    x_max = torch.max(x, dim, True).values
    ret = torch.log(torch.mean(torch.exp(x - x_max), dim, True)) + x_max
    print("???", loss + ret)
    if not keepdims:
        ret = torch.mean(ret, dim=dim)
    return ret
