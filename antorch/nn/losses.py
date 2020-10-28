import antorch

class _Loss:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MSELoss(_Loss):
    def forward(self, inputs, targets):
        return antorch.sum((inputs - targets) ** 2) / len(targets)
