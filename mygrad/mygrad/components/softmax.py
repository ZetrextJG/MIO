from collections.abc import Iterable
from mygrad.components.base import Component
import numpy as np

from mygrad.parameters import Parameter

EPSILON = 1e-8


def softmax_stable(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1).reshape(-1, 1)


def softmax_original(x):
    intermid = np.exp(x)
    sums = np.sum(intermid, axis=1)
    softmax = intermid / sums.reshape(-1, 1)
    return softmax


class Softmax(Component):
    def __init__(self, version="original"):
        if version == "original":
            self.softmax_function = softmax_original
        elif version == "stable":
            self.softmax_function = softmax_stable

    def forward(self, x: np.ndarray) -> np.ndarray:
        softmax = self.softmax_function(x)
        if self.training:
            self.last_softmax = softmax
        return softmax

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the softmax layer.

        For a single vector x the jacobian of the softmax layer can be calculated as:
        s = softmax(x) # s is a row vector
        jacobian = diag(s) - s^T . s

        For a batch this becomes more complicated. In order no to calculated the
        jacobian for each element in the batch we can use the previous gradient
        and some multiplication tricks in order to calculate the batch of gradients
        needed for the backpropagation.
        """
        assert self.training, "Backward should not be called in evaluation mode"
        assert self.last_softmax is not None, "Forward must be called before backward"

        b = np.einsum("ij,ij->i", grad, self.last_softmax).reshape(-1, 1)
        return (grad - b) * self.last_softmax

    def next_dim(self, dim):
        return dim

    def parameters(self) -> Iterable[Parameter]:
        return iter([])

    def zero_grad(self):
        self.last_softmax = None

    def __str__(self):
        return "Softmax()"
