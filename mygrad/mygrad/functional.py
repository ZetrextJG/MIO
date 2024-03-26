from typing import Optional
import numpy as np


def identity(x: np.ndarray) -> np.ndarray:
    return x


def linear(x: np.ndarray, A: np.ndarray, b: Optional[np.ndarray] = None):
    """Applies the linear(affine) transformation xA^T +b"""
    inter = np.matmul(x, A.T)
    if b is not None:
        return inter + b
    else:
        return inter


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def mse(x1: np.ndarray, x2: np.ndarray, axis=0) -> np.ndarray:
    return np.mean((x1 - x2) ** 2, axis=axis)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean(y_pred == y_true)


def onehot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y].reshape(-1, num_classes)


def onehot_decode(y: np.ndarray) -> np.ndarray:
    return np.argmax(y, axis=1)


def fscore_onehot(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the macro F-score for the multiclass classification.
    Assumes that the inputs are of shape (batch_size, num_classses).
    Assuess that the labels are one-hot encoded.
    """

    tp = np.sum(y_pred * y_true, axis=0)
    precision = tp / np.sum(y_pred, axis=0)
    recall = tp / np.sum(y_true, axis=0)
    fscores = 2 * precision * recall / (precision + recall)
    return np.mean(fscores)  # Macro F-score


def fscore(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    y_pred = onehot_encode(y_pred, num_classes)
    y_true = onehot_encode(y_true, num_classes)
    return fscore_onehot(y_pred, y_true)
