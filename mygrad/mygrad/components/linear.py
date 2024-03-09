from typing import Callable, Dict, Iterator, Optional, Literal
import numpy as np
import mygrad.components as mc
import mygrad.functional as ff
from mygrad.parameters import Parameter


# Initialization functions
def uniform(input_size: int, output_size: int):
    """Returns values from uniform distribution between -1 and 1"""
    return 2 * np.random.rand(input_size, output_size) - 1


def normal(input_size: int, output_size: int):
    """Returns values from a standard normal distribution"""
    return np.random.randn(input_size, output_size)


def xavier_method(input_size: int, output_size: int):
    """Returns values initialized using the Xavier method."""
    return uniform(input_size, output_size) * np.sqrt(6 / (input_size + output_size))


def he_method(input_size: int, output_size: int):
    """Returns values initialized using the He method."""
    return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)


INIT_METHODS: Dict[str, Callable[[int, int], np.ndarray]] = {
    "uniform": uniform,
    "normal": normal,
    "xavier": xavier_method,
    "he": he_method,
}
INIT_METHODS_STR = Literal["uniform", "normal", "xavier", "he"]


# Linear layer
class Linear(mc.Component):
    input_size: int
    output_size: int

    def __init__(
        self,
        input_size: int,
        output_size: int,
        init: INIT_METHODS_STR = "xavier",
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ):
        self.output_size = output_size

        if weights is None:
            f = INIT_METHODS[init]
            weights = f(input_size, output_size).T

        if bias is None:
            bias = np.zeros(output_size)

        self.W = Parameter(weights)
        self.b = Parameter(bias)
        self.parameters_ = [self.W, self.b]

    def next_dim(self, dim: int) -> int:
        return self.output_size

    def parameters(self) -> Iterator[Parameter]:
        for param in self.parameters_:
            yield param

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        sigma = ff.linear(x, self.W.data, self.b.data)
        return sigma

    def backward(self, grad: np.ndarray) -> np.ndarray:
        n = len(self.last_x)
        backward_grad = grad @ self.W.data

        self.W.grad += (grad.T @ self.last_x) / n
        self.b.grad += np.mean(grad, axis=0)

        return backward_grad

    def zero_grad(self):
        for param in self.parameters_:
            param.zero_grad()
