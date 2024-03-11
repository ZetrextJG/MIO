from collections.abc import Iterator
from typing import Literal, Optional
import numpy as np
from abc import ABC, abstractmethod

from mygrad.components import Component
from mygrad.parameters import Parameter


class ActivationFunction(Component, ABC):
    activation_grad: Optional[np.ndarray]

    def __init__(self):
        self.activation_grad = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.activation_grad is not None, "The activation gradient is not set"
        return grad * self.activation_grad

    def next_dim(self, dim: int) -> int:
        return dim

    def zero_grad(self):
        assert self.activation_grad is not None, "The activation gradient is not set"
        self.activation_grad = None

    def parameters(self) -> Iterator[Parameter]:
        return iter([])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __str__(self):
        return self.__class__.__name__ + "()"


class Identity(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activation_grad = np.ones_like(x)
        return x


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        sig = 1 / (1 + np.exp(-x))
        self.activation_grad = sig * (1 - sig)
        return sig


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        tan = np.tanh(x)
        self.activation_grad = 1 - tan**2
        return tan


class ReLU(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activation_grad = (x > 0) * 1
        return np.maximum(0, x)


class LeakyReLU(ActivationFunction):
    leak: float

    def __init__(self, leak: float = 0.01):
        self.leak = leak

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activation_grad = (x > 0) * 1 + self.leak * (x <= 0)
        return np.maximum(self.leak * x, x)


class ThresholdJump(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        value = (x > 0) * 1
        self.activation_grad = (
            value  # HACK: This in not really the gradient, but it will do
        )
        return value


ACTIVATION_METHODS: dict[str, type[ActivationFunction]] = {
    "identity": Identity,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "threshold": ThresholdJump,
}
ACTIVATION_METHODS_STR = Literal["identity", "tanh", "sigmoid", "relu", "threshold"]
