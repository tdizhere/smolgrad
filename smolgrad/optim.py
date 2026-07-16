import math
class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        raise NotImplementedError
class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
class Momentum(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.velocity = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):

            self.velocity[i] = (
                self.momentum * self.velocity[i]
                + p.grad
            )

            p.data -= self.lr * self.velocity[i]
    

class RMSProp(Optimizer):

    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(params)

        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.v = [0.0 for _ in self.params]

    def step(self):

        for i, p in enumerate(self.params):

            g = p.grad

            self.v[i] = (
                self.beta * self.v[i]
                + (1 - self.beta) * (g * g)
            )

            p.data -= self.lr * g / (math.sqrt(self.v[i]) + self.eps)
class Adam(Optimizer):

    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):
        super().__init__(params)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]

        self.t = 0

    def step(self):

        self.t += 1

        for i, p in enumerate(self.params):

            g = p.grad

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1 - self.beta1) * g
            )

            self.v[i] = (
                self.beta2 * self.v[i]
                + (1 - self.beta2) * (g * g)
            )

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= (
                self.lr
                * m_hat
                / (math.sqrt(v_hat) + self.eps)
            )