from typing import Iterator
import numpy as np

from mygrad.optimizers import Optimizer
from mygrad.parameters import Parameter


class GradientDescent(Optimizer):
    learning_rate: float

    def __init__(self, params: Iterator[Parameter], learning_rate: float):
        self.params = list(params)
        self.learning_rate = learning_rate

    def step(self):
        for param in self.params:
            param.data -= self.learning_rate * param.grad
