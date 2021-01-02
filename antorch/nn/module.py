from collections import OrderedDict
import math
import antorch

class Module:
    def __init__(self):
        self.training = True
        self._parameters = []
        self._buffers = OrderedDict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        for params in self.parameters():
            params.zero_grad()

    def parameters(self):
        if self._parameters: return self._parameters
        
        for i, p in self.__dict__.items():
            if isinstance(p, antorch.Tensor):
                self._parameters.append(p)
            if isinstance(p, Module):
                self._parameters.extend(p.parameters())

        return self._parameters

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()

        self._modules = tuple(args)

    def __iter__(self):
        yield from self._modules

    def __getitem__(self, idx):
        return self._modules[idx]

    def forward(self, x):
        for module in self:
            x = module(x)
        return x

    def parameters(self):
        if self._parameters == []:
            for module in self:
                self._parameters.extend(module.parameters())

        return self._parameters

