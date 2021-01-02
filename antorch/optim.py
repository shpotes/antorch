import numpy as np

class SGD:
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        self.defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': 0
        }

        self.params = params
        self.momentum = [np.zeros_like(p.grad) for p in params] if momentum != 0 else []
        self.global_step = 0

    def step(self):
        weight_decay = self.defaults['weight_decay']
        lr = self.defaults['lr']
        
        momentum = self.defaults['momentum']

        for idx, p in enumerate(self.params):
            d_p = p.grad

            #if weight_decay != 0:
            #    d_p += p.grad * weight_decay

            if momentum != 0:
                buf = self.momentum[idx]
                np.add(momentum * buf, d_p, out=buf)
                d_p = buf

            
            np.add(p.data, - lr * d_p, out=p.data)

        self.global_step += 1
