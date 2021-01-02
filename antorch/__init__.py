__version__ = '0.1.0'

from .core import (
    Tensor,
    zeros,
    ones,
    randn,
    rand,
)

from .ops import (
    exp,
    where,
    sum,
    mean,
    log
)

from . import nn, optim
