from typing import Callable, Union
import numpy as np

def _unbroadcast(inp, out):
    i_shape = inp.shape

    if inp.ndim < out.ndim:
        i_shape = np.expand_dims(
            inp,
            list(range(out.ndim - inp.ndim))
        ).shape

    axis = [
        i for i, (i_ax, o_ax) in enumerate(zip(i_shape, out.shape))
        if i_ax != o_ax
    ]

    return tuple(axis)

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, float) or isinstance(data, int):
            data = [data]
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError

        self.dtype = np.float32
        self.shape = data.shape
        self.ndim = data.ndim
        self.data = data.astype(self.dtype)
        self.grad = np.zeros_like(data, dtype=self.dtype)

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def view(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad = self.grad.reshape(*shape)

        out._backward = _backward

        return out

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=self.dtype)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            s_axis = _unbroadcast(self.data, out.data)
            o_axis = _unbroadcast(other.data, out.data)

            self.grad += out.grad.sum(s_axis).reshape(*self.shape)
            other.grad += out.grad.sum(o_axis).reshape(*other.shape)

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            s_axis = _unbroadcast(self.data, out.data)
            o_axis = _unbroadcast(other.data, out.data)

            self.grad += (out.grad * other.data).sum(s_axis).reshape(self.shape)
            other.grad += (out.grad * self.data).sum(o_axis).reshape(other.shape)

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __pow__(self, power: Union[float, int]):
        out = Tensor(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        seen = set()

        def topo_sort(node: Tensor):
            if node not in seen:
                seen.add(node)
                for child in node._prev:
                    topo_sort(child)
                topo.append(node)

        topo_sort(self)

        #print(topo)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            #print(node.shape, node.grad.shape, node._op)
            node._backward()

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __getitem__(self, item):
        return self.data[item]

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __str__(self) -> str:
        return f'Tensor(data={self.data}, grad_f={self._op})'

    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, grad_f={self._op})'

    def __eq__(self, other) -> bool:
        return str(self) == str(other) and str(self._prev) == str(other._prev)

    def __hash__(self) -> int:
        return hash(str(self))

    def __len__(self) -> int:
        return len(self.data)

    def numpy(self):
        return self.data

    def detach(self):
        return Tensor(self.data)



def zeros(shape):
    return Tensor(np.zeros(shape))

def ones(shape):
    return Tensor(np.ones(shape))

def randn(shape, mu=0, sigma=1):
    seed = sigma * np.random.randn(*shape) + mu
    return Tensor(seed)

def rand(shape, lower, upper):
    seed = np.random.uniform(lower, upper, shape)
    return Tensor(seed)
