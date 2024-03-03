from typing import Any
import numpy as np

from mygrad.preprocessors.base import Preprocessor


class MinMaxScaler(Preprocessor):
    _mins: np.floating[Any]
    _maxs: np.floating[Any]

    def __init__(self):
        self._mins = 0
        self._maxs = 1
        self.was_fit = False

    def fit(self, x: np.ndarray):
        if self.was_fit:
            raise RuntimeError("MinMaxScaler has already been fit")

        self._mins = np.min(x, axis=0)
        self._maxs = np.max(x, axis=0)
        if np.allclose(self._mins, self._maxs):
            raise ValueError(
                "MinMaxScaler: min and max are equal for at least one feature"
            )

        self.was_fit = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.was_fit:
            raise RuntimeError("MinMaxScaler has not been fit")

        return (x - self._mins) / (self._maxs - self._mins)

    def reverse(self, x: np.ndarray) -> np.ndarray:
        if not self.was_fit:
            raise RuntimeError("MinMaxScaler has not been fit")

        return x * (self._maxs - self._mins) + self._mins

    def __repr__(self):
        return f"MinMaxScaler(min={self._mins}, max={self._maxs})"


class StandardScaler(Preprocessor):
    _means: np.floating[Any]
    _stds: np.floating[Any]

    def __init__(self):
        self._means = 0
        self._stds = 1
        self.was_fit = False

    def fit(self, x: np.ndarray):
        if self.was_fit:
            raise RuntimeError("StandardScaler has already been fit")

        self._means = np.mean(x, axis=0)
        self._stds = np.std(x, axis=0)

        if np.any(self._stds == 0):
            raise ValueError("StandardScaler: at least one feature has zero variance")

        self.was_fit = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.was_fit:
            raise RuntimeError("StandardScaler has not been fit")

        return (x - self._means) / self._stds

    def reverse(self, x: np.ndarray) -> np.ndarray:
        if not self.was_fit:
            raise RuntimeError("StandardScaler has not been fit")

        return x * self._stds + self._means

    def __repr__(self):
        return f"StandardScaler(mean={self._means}, std={self._stds})"
