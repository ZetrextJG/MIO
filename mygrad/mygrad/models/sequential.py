import numpy as np

from mygrad.models.base import Model


class Sequential(Model):
    def __init__(self, *components):
        self.components = components

    def forward(self, x: np.ndarray) -> np.ndarray:
        for component in self.components:
            x = component.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for component in reversed(self.components):
            grad = component.backward(grad)
        return grad

    def reset_grad(self):
        for component in self.components:
            component.reset_grad()
