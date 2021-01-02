import math
import antorch
import antorch.nn as nn
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        weight_bound = math.sqrt(3 / in_features)

        self.weight = antorch.rand(
            shape=(in_features, out_features),
            lower=-weight_bound, upper=weight_bound
        )

        if bias:
            bias_bound = 1 / math.sqrt(in_features)
            self.bias = antorch.rand(out_features, -bias_bound, bias_bound)
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight

        if self.bias is not None:
            out += self.bias

        return out
