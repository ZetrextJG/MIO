import numpy as np
from abc import ABC, abstractmethod


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
    def reset_grad(self) -> np.ndarray:
        ...
