from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np

from mygrad.parameters import Parameter


class Component(ABC):
    def __init__(self) -> None:
        self.training: bool = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Does the forward pass of the component"""
        ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Does the backward pass of the component with gradient backpropagated from the next layer"""
        ...

    @abstractmethod
    def next_dim(self, dim: int) -> int:
        """Returns the output dimension of the component given the input dimension"""
        ...

    @abstractmethod
    def parameters(self) -> Iterator[Parameter]:
        """Returns an iterator over the parameters of the component"""
        ...

    @abstractmethod
    def zero_grad(self):
        """Clears the gradients of the parameters of the component"""
        ...


class FixedDimension:
    input_size: int
    output_size: int

    def next_dim(self, dim: int) -> int:
        assert dim == self.input_size
        return self.output_size
