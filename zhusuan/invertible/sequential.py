from __future__ import absolute_import
from __future__ import division
import torch

from zhusuan.invertible.base import RevNet


class RevSequential(RevNet):
    def __init__(self, layers):
        super(RevSequential, self).__init__()
        self.layers = layers
        for flow in self.layers:
            assert isinstance(flow, RevNet)

    def _forward(self, *inputs, **kwargs):
        logdet_items = []
        x = None
        for flow in self.layers:
            x = flow(*inputs, reverse=False, **kwargs)
            assert isinstance(x, tuple)
            assert len(x) >= 2
            if x[-1] is not None:
                logdet_items.append(x[-1])
            if isinstance(x[0], tuple):
                x = x[0]
            else:
                x = x[:len(x) - 1]
            assert isinstance(x, tuple)

        return x, sum(logdet_items) if logdet_items else torch.zeros([])

    def _inverse(self, *inputs, **kwargs):
        logdet_item = []
        y = None
        for flow in reversed(self.layers):
            y = flow(*y, reverse=True, **kwargs)
            assert isinstance(y, tuple)
            assert len(y) > 2
            if y[-1] is not None:
                logdet_item.append(y[-1])
            if isinstance(y[0], tuple):
                y = y[0]
            else:
                y = y[:len(y) - 1]
            assert isinstance(y, tuple)
        return y, sum(logdet_item) if logdet_item else torch.zeros([])
