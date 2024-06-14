from typing import Iterator
import numpy as np

from mygrad.components.base import Component
from mygrad.parameters import Parameter


class Dropout(Component):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.p
            return x * self.mask / (1 - self.p)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.training, "Backward should not be called in evaluation mode"
        assert self.mask is not None, "Forward must be called before backward"
        return grad * self.mask

    def parameters(self) -> Iterator[Parameter]:
        return iter([])

    def next_dim(self, dim: int) -> int:
        return dim

    def zero_grad(self):
        self.mask = None

    def __str__(self):
        return f"Dropout(p={self.p})"
