from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        ...

    def both(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (self.forward(x), self.grad(x))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Identity(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)

    def both(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sig = self.forward(x)
        return (sig, sig * (1 - sig))


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

    def both(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tanh = self.forward(x)
        return (tanh, 1 - tanh**2)


class ReLU(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(int)
