from __future__ import absolute_import
from __future__ import division
import torch.nn as nn

__all__ = [
    "RevNet"
]


class RevNet(nn.Module):
    """
    An abc of reversible network，every subclass should implement both ``_forward`` and ``_inverse`` abstract method.

    """
    def _forward(self, *inputs, **kwargs):
        raise NotImplementedError()

    def _inverse(self, *inputs, **kwargs):
        raise NotImplementedError()

    def forward(self, *inputs, reverse=False, **kwargs):
        """
        when using ``model.forward(x, reverse=False)`` process going with ``_forward(x)``,
        when using ``model.forward(x, reverse=True)`` process going with ``_inverse(x)``.
        """
        if not reverse:
            return self._forward(*inputs, **kwargs)
        else:
            return self._inverse(*inputs, **kwargs)
