from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import mygrad.functional as ff


class Loss(ABC):
    @abstractmethod
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        ...

    def both(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (self.value(y_pred, y_true), self.grad(y_pred, y_true))


class DummyAbsoluteDifferenceLoss(Loss):
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(y_pred - y_true), axis=0)

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """This is not formally the gradient by it will do"""
        return y_pred - y_true


class AbsoluteDifferenceLoss(Loss):
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(y_pred - y_true), axis=0)

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true)


class MeanSquareErrorLoss(Loss):
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return ff.mse(y_pred, y_true, axis=0)

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)


class LogCoshLoss(Loss):
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.log(np.cosh(y_pred - y_true)).sum()

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.tanh(y_pred - y_true)
