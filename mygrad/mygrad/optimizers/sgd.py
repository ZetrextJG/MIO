from typing import Iterator, Optional
import numpy as np

from mygrad.optimizers import Optimizer
from mygrad.parameters import Parameter


class SGD(Optimizer):
    learning_rate: float

    def __init__(
        self,
        params: Iterator[Parameter],
        learning_rate: float,
        weight_decay: float = 0,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
    ):
        self.params = list(params)

        self.momentums_ = [np.zeros_like(param.data) for param in self.params]

        # TODO: Might need optimization later
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov

        self.step_count = 0

    def step(self):
        for param, momentum in zip(self.params, self.momentums_):
            change = param.grad.copy()

            if self.weight_decay != 0:
                change += self.weight_decay * param.data

            if self.momentum != 0:
                if self.step_count > 0:
                    momentum += momentum * self.momentum + (1 - self.dampening) * change
                else:
                    momentum = change

                if self.nesterov:
                    change += self.momentum * momentum
                else:
                    change = momentum

            param.data -= self.learning_rate * change
            self.step_count += 1
