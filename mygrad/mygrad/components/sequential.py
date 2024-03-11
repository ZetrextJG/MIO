from typing import Iterator
from functools import cache
import numpy as np

from mygrad.parameters import Parameter
from mygrad.components import Component
from tests.test_utils import indent_lines


class Sequential(Component):
    def __init__(self, *components: Component):
        self.components = components

    def forward(self, x: np.ndarray) -> np.ndarray:
        for component in self.components:
            x = component.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for component in reversed(self.components):
            grad = component.backward(grad)
        return grad

    @cache
    def next_dim(self, dim: int) -> int:
        for component in self.components:
            dim = component.next_dim(dim)
        return dim

    def zero_grad(self):
        for component in self.components:
            component.zero_grad()

    def parameters(self) -> Iterator[Parameter]:
        for component in self.components:
            yield from component.parameters()

    def __str__(self):
        body = ",\n".join([str(comp) for comp in self.components])
        indeted_body = indent_lines(body)
        return f"Sequential(\n{indeted_body}\n)"
