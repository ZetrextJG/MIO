import numpy as np

from mygrad.components.base import Component


class Dropout(Component):
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.p
            return x * self.mask / (1 - self.p)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.training, "Backward should not be called in evaluation mode"
        assert self.mask is not None, "Forward must be called before backward"
        return grad * self.mask

    def next_dim(self, dim: int) -> int:
        return dim

    def zero_grad(self):
        self.mask = None

    def __str__(self):
        return f"Dropout(p={self.p})"
