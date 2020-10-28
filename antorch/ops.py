import numpy as np
from .core import Tensor, zeros, ones

def exp(x: Tensor) -> Tensor:
    out = Tensor(np.exp(x), (x,), 'exp')

    def _backward():
        x.grad += np.exp(x) * out.grad

    out._backward = _backward

    return out

def where(cond: np.ndarray, x: Tensor, y: Tensor) -> Tensor:
    x = x if isinstance(x, Tensor) else Tensor(x)
    y = y if isinstance(y, Tensor) else Tensor(y)
    
    out = Tensor(np.where(cond, x.data, y.data), (x, y), 'where')

    def _backward():
        x.grad += np.where(cond, out.grad, 0)
        y.grad += np.where(cond, 0, out.grad)

    out._backward = _backward

    return out


def sum(x: Tensor, axis=None) -> Tensor:
    if axis is not None:
        raise ValueError

    out = Tensor(x.data.sum().reshape(1), (x,), 'sum')

    def _backward():
        x.grad += out.grad.sum()

    out._backward = _backward

    return out
