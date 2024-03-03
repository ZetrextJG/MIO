import numpy as np
from abc import ABC, abstractmethod

from mygrad.utils import Component


class Model(Component, ABC):
    components: tuple[Component, ...]

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Does the forward pass of the model"""
        ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes gradient for parameters of the model and returns the gradient for the input"""
        ...

    @abstractmethod
    def reset_grad(self):
        """Resets the gradients for the parameters of the model"""
        ...
