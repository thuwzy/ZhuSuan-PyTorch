import torch
import torch.nn as nn

from zhusuan.transforms.invertible import InvertibleTransform

class Scaling(InvertibleTransform):
    """
    The scaling layer described in NICE paper :cite:`scaling-dinh2015nice`, which compute the following process and its inverse.

    .. math::

        \\begin{bmatrix}
        S_1 &      &        &     

            & S_2  &        &      

            &      & \ddots &       

            &      &        &  S_D   
        \end{bmatrix}
        \\begin{bmatrix}
        h_{i - 1, 1} 

        h_{i - 1, 2} 

        \\vdots 

        h_{i - 1, D}
        \end{bmatrix}
        =
        \\begin{bmatrix}
        h_{i, 1} 

        h_{i , 2} 

        \\vdots 
        
        h_{i, D}
        \end{bmatrix}

    :param n_dim: The dim of the Var to be transformed.

    .. rubric:: References

    .. bibliography:: ../refs.bib
        :style: unsrtalpha
        :keyprefix: scaling-
    """
    def __init__(self, n_dim, device = torch.device('cpu')):
        super().__init__()
        self.device = device
        # self.log_scale = nn.init.constant(shape=[1, n_dim], dtype=torch.float32)
        self.log_scale = torch.zeros(1, n_dim, dtype=torch.float32).to(self.device) #!TODO

    def _forward(self, x, **kwargs):
        log_detJ = self.log_scale.clone()
        x *= torch.exp(self.log_scale)
        return x, log_detJ

    def _inverse(self, z, **kwargs):
        z *= torch.exp(-self.log_scale)
        return z, None
