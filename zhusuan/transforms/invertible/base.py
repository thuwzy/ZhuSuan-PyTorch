from zhusuan.transforms.base import Transform

class InvertibleTransform(Transform):
    """
    Base class for invertible transforms in normalizing flows.
    Each instance of :class:`~zhusuan.transforms.invertible.base.InvertibleTransform` should
    implement both :meth:`_forward` and :meth:`_inverse`
    """
    def __init__(self):
        super().__init__()
        self.is_invertible = True
    