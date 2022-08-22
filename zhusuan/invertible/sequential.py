from __future__ import absolute_import
from __future__ import division
import torch

from zhusuan.invertible.base import RevNet


class RevSequential(RevNet):
    """
    the RevSequential provide a invertible transform which contain a list of instance of RevNet.
    when forward passing with ``reverse=False``, the input ``x`` goes through every RevNet in the list also with
    ``reverse=False`` *from begin to end* , when forward passing with ``reverse=True``, input ``x`` goes through
    every in the list also with ``reverse=True`` *from end to begin*.

    :param layers: a list of RevNet instance.
    """
    def __init__(self, layers):
        super(RevSequential, self).__init__()
        for flow in layers:
            assert isinstance(flow, RevNet)
        self.layers = torch.nn.ModuleList(layers)

    def _forward(self, x, **kwargs):
        logdet_items = []
        for flow in self.layers:
            x, log_det = flow(x, reverse=False, **kwargs)
            if log_det is not None:
                logdet_items.append(log_det)

        return x, sum(logdet_items) if logdet_items else torch.zeros([])

    def _inverse(self, y, **kwargs):
        logdet_item = []
        for flow in reversed(self.layers):
            y, log_det = flow(y, reverse=True, **kwargs)
            if log_det is not None:
                logdet_item.append(log_det)

        return y, sum(logdet_item) if logdet_item else torch.zeros([])
