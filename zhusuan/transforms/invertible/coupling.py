import torch
import torch.nn as nn

from zhusuan.transforms.invertible import InvertibleTransform

def get_coupling_mask(n_dim, n_channel, n_mask, split_type='ChessBoard', device = torch.device('cpu')):
    """
    Mask generator for coupling layers.

    :param n_dim: The number of the dim which to be divided.
    :param n_channel: The channel of the Var.
    :param n_mask: The number of masks Var to be generated.
    :param split_type: The way to divide the var in coupling layer. Only the default Chessboard(or OddEven) way is supported now.
    :return: A list of Var, which will be applied to the original Var on each coupling layer.
    """
    with torch.no_grad():
        masks = []
        if split_type == 'ChessBoard':
            if n_channel == 1:
                mask = torch.arange(0, n_dim, dtype=torch.float32) % 2
                for i in range(n_mask):
                    masks.append(mask.to(device))
                    mask = (1. - mask)
        else:
            raise NotImplementedError()
        return masks


class AdditiveCoupling(InvertibleTransform):
    """
    Additive coupling layer.
    Computes the following process and its inverse process:

    .. math::

        &\mathbf x_{I_1}, \mathbf x_{I_2} = \\text{split}(\mathbf h_{i - 1})

        &\mathbf y_{I_1} = \mathbf x_{I_1}, \ \ \mathbf y_{I_2} = \mathbf x_{I_2} + m(\mathbf x_{I_1}) 

        &\mathbf h_{i} = f_{i}(\mathbf h_{i - 1}) = \\text{concat}(\mathbf y_{I_1}, \mathbf y_{I_2})

    The layer provides a default fully connected layers which are similar to NICE paper :cite:`coupling-dinh2015nice`, or you can
    use any customize network using the `inner_nn` parameter.

    :param in_out_dim: The dim of the Var to be transformed.
    :param mid_dim: The middle dim of the default net.
    :param hidden: The number of hidden layers of the default net/
    :mask: The mask Var acting on this coupling layer.
    :inner_nn: The customize inner network.
    
    .. rubric:: References

    .. bibliography:: ../refs.bib
        :style: unsrtalpha
        :keyprefix: coupling-
    """
    def __init__(self, in_out_dim=-1, mid_dim=-1, hidden=-1, mask=None, inner_nn=None):
        super().__init__()
        if inner_nn is None:
            self.nn = []
            self.nn += [nn.Linear(in_out_dim, mid_dim),
                        nn.ReLU()]
            for _ in range(hidden - 1):
                self.nn += [nn.Linear(mid_dim, mid_dim),
                            nn.ReLU()]
            self.nn += [nn.Linear(mid_dim, in_out_dim)]
            self.nn = nn.Sequential(*self.nn)
        else:
            self.nn = inner_nn
        self.mask = mask
    
    def _forward(self, x, **kwargs):
        #print("x type:", x.device, "mask type:", self.mask.device)
        x1, x2 = self.mask * x, (1 - self.mask) * x
        shift = self.nn(x1)
        z1, z2 = x1, x2 + shift * (1. - self.mask)
        return z1 + z2, None
    
    def _inverse(self, z, **kwargs):
        z1, z2 = self.mask * z, (1 - self.mask) * z
        shift = self.nn(z1)
        x1, x2 = z1, z2 - shift * (1. - self.mask)
        return x1 + x2, None


