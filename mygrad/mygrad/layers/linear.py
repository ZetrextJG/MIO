from typing import Callable, Iterator, Optional, Tuple
import numpy as np
from mygrad.activations import ActivationFunction, Tanh
from mygrad.layers.base import Layer
import mygrad.functional as ff
from mygrad.parameters import Parameter


def uniform(output_size: int, input_size: int):
    """Returns values from uniform distribution between -1 and 1"""
    return 2 * np.random.rand(input_size, output_size) - 1


def normal(output_size: int, input_size: int):
    """Returns values from a standard normal distribution"""
    return np.random.randn(input_size, output_size)


def xavier_method(output_size: int, input_size: int):
    """Returns values initialized using the Xavier method."""
    return uniform(input_size, output_size) * np.sqrt(6 / (input_size + output_size))


def he_method(output_size: int, input_size: int):
    """Returns values initialized using the He method."""
    return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)


class Linear(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        init_function: Callable[[int, int], np.ndarray] = xavier_method,
        activation_function: ActivationFunction = Tanh(),
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ):
        self.input_size = input_size
        self.output_size = output_size

        if weights is None:
            weights = init_function(output_size, input_size)

        if bias is None:
            bias = np.zeros(output_size)

        self.W = Parameter(weights)
        self.b = Parameter(bias)
        self.parameters_ = [self.W, self.b]

        self.activation_function = activation_function
        self.activation_grad = np.zeros(output_size)

    def parameters(self) -> Iterator[Parameter]:
        for param in self.parameters_:
            yield param

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        sigma = ff.linear(x, self.W.data, self.b.data)
        value, grad = self.activation_function.both(sigma)
        self.activation_grad = grad
        return value

    def backward(self, grad: np.ndarray) -> np.ndarray:
        n = len(self.last_x)
        z = grad * self.activation_grad
        backward_grad = z @ self.W.data

        self.W.grad += (z.T @ self.last_x) / n
        self.b.grad += np.mean(z, axis=0)

        return backward_grad

    def reset_grad(self):
        for param in self.parameters_:
            param.zero_grad()
