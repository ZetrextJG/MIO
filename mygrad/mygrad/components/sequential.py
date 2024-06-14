from typing import Iterator
from functools import cache
import numpy as np
from mygrad.components.base import FixedDimension

from mygrad.parameters import Parameter
from mygrad.components import Component
from mygrad.utils import indent_lines


class Sequential(FixedDimension, Component):
    def __init__(self, *components: Component):
        self.components = components

        first = components[0]
        assert isinstance(
            first, FixedDimension
        ), "First component must have a fixed dimension"
        self.input_size = first.input_size
        self.output_size = self.next_dim(self.input_size)

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

    def train(self) -> None:
        super().train()
        for component in self.components:
            component.train()

    def eval(self) -> None:
        super().eval()
        for component in self.components:
            component.eval()
