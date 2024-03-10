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
