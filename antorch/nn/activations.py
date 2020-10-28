import antorch
from .module import Module

class ReLU(Module):
    def forward(self, x):
        return antorch.where(x.data > 0, x, antorch.zeros(x.shape))

class LeakyReLU(Module):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, x):
        return antorch.where(x.data > 0, x, self.alpha * x)

class Tanh(Module):
    def forward(self, x):
        return (antorch.exp(x) - antorch.exp(-x)) / (antorch.exp(x) + antorch.exp(-x))

class Sigmoid(Module):
    def forward(self, x):
        return (1 + antorch.exp(-x)) ** -1
