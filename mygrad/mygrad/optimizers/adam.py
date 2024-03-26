import numpy as np
from collections.abc import Iterator
from mygrad.parameters import Parameter
from mygrad.optimizers.base import Optimizer


EPSILON = 1e-8


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
    ):
        self.params = list(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros_like(p.data) for p in self.params]  # First moment
        self.v = [np.zeros_like(p.data) for p in self.params]  # Second moment

    def step(self):
        for i, param in enumerate(self.params):
            grad = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_hat = self.m[i] / (1 - self.beta1)
            v_hat = self.v[i] / (1 - self.beta2)
            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

    def end_epoch(self):
        for i in range(len(self.params)):
            self.m[i] *= 0
            self.v[i] *= 0
