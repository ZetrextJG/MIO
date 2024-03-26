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
        n = y_pred.shape[0]
        return 2 * (y_pred - y_true) / n


class LogCoshLoss(Loss):
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.log(np.cosh(y_pred - y_true)).mean()

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_pred.shape[0]
        return np.tanh(y_pred - y_true) / n


EPSILON = 1e-12


class BinaryCrossEntropy(Loss):
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the binary cross entropy.
        The input vectors are assumed to be of shape (batch_size x 1) vectors.
        """
        losses = (-1) * (
            (1 - y_true) * np.log(1 - y_pred + EPSILON)
            + y_true * np.log(y_pred + EPSILON)
        )
        return losses.mean()

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        g = (1 - y_true) / (1 - y_pred + EPSILON) - y_true / (y_pred + EPSILON)
        return g


class CategorialCorssEntropy(Loss):
    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the categorial cross entropy.
        Input vectors are assumed to be of shape (batch_size x num_classes vectors).
        """
        losses = -np.sum(y_true * np.log(y_pred + EPSILON), axis=1)
        return np.mean(losses)

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return -y_true / (y_pred + EPSILON)
