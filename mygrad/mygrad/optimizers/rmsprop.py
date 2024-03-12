import numpy as np
from collections.abc import Iterator
from mygrad.parameters import Parameter
from mygrad.optimizers.base import Optimizer


class RMSProp(Optimizer):
    learning_rate: float

    def __init__(
        self, params: Iterator[Parameter], learning_rate: float, beta: float = 0.9
    ):
        self.params = list(params)
        self.learning_rate = learning_rate
        self.beta = beta

        self.squared_gradients = [np.zeros_like(param.grad) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.squared_gradients[i] = (
                self.beta * self.squared_gradients[i] + (1 - self.beta) * param.grad**2
            )
            change = param.grad / (np.sqrt(self.squared_gradients[i]) + 1e-8)
            param.data -= self.learning_rate * change
