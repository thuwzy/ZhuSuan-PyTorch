import torch.nn as nn

class Transform(nn.Module):
    """
    Base class for Transforms.
    """
    def __init__(self):
        super().__init__()
        self.is_invertible = True
    
    def _forward(self, *args, **kwargs):
        """
        Forward transform.
        Compute :math:`x \mapsto z` and the log_abs determinant jacobian term.
        """
        raise NotImplementedError()
    
    def _inverse(self, *args, **kwargs):
        """
        Inverse transform.
        Compute :math:`z \mapsto x`.
        """
        raise NotImplementedError()

    def forward(self, *args, inverse=False, **kwargs):
        """
        Do forward and inverse transform.
    
        * **Forward transform**: Compute :math:`x \mapsto z` and the log_abs determinant jacobian term.
        * **Inverse transform**: Compute :math:`z \mapsto x`.

        :param inverse: A Bool. Indicates whether execute the forward transform or the inverse transform.
        """
        if not inverse:
            return self._forward(*args, **kwargs)
        else:
            return self._inverse(*args, **kwargs)