from abc import ABC, abstractmethod
import numpy as np

from mygrad.layers.parameters import Parameter
from mygrad.utils import Component


class Layer(Component, ABC):
    parameters: list[Parameter]

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Does the forward pass of the layer"""
        ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes gradient for parameters of the layer and returns the gradient for the input"""
        ...

    @abstractmethod
    def reset_grad(self):
        """Resets the gradients for the parameters of the layer"""
        ...
