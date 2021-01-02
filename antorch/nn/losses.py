import antorch

class _Loss:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class MSELoss(_Loss):
    def forward(self, inputs, targets):
        return antorch.mean((inputs - targets) ** 2)

class BCELoss(_Loss):
    def forward(self, inputs, targets):
        return -antorch.mean(
            targets * antorch.log(inputs) + (-targets + 1) * antorch.log(-inputs + 1)
        )
