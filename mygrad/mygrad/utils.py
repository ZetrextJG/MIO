from typing import Generator, Iterator
import numpy as np
from abc import ABC, abstractmethod

from mygrad.parameters import Parameter


class Component(ABC):
    input_size: int
    output_size: int

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def parameters(self) -> Iterator[Parameter]:
        ...

    @abstractmethod
    def zero_grad(self) -> np.ndarray:
        ...
