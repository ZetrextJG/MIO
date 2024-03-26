from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from numpy.lib import math


class Dataloader(ABC):
    batch_size: int

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        ...


class NumpyRegressionDataloader(Dataloader):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False
    ):
        assert len(x.shape) == 2, "x must be a 2D array"
        assert len(y.shape) == 2, "x must be a 2D array"
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
        assert y.shape[1] == 1, "y must be a 2D array with a single column"

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_length = x.shape[0]

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def shuffle_data(self) -> None:
        p = np.random.permutation(len(self.x))
        self.x = self.x[p, :]
        self.y = self.y[p, :]

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            self.shuffle_data()

        iterations = math.ceil(len(self.x) / self.batch_size)
        for i in range(0, iterations):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, self.sample_length)
            yield (
                self.x[start_index:end_index].copy(),
                self.y[start_index:end_index].copy(),
            )


class NumpyClassificationDataloader(Dataloader):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        is_one_hot: bool = False,
    ):
        assert len(x.shape) == 2, "x must be a 2D array"
        assert len(y.shape) == 2, "x must be a 2D array"
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
        if not is_one_hot:
            assert y.shape[1] == 1, "y must be a 2D array with a single column"

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_length = x.shape[0]

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def shuffle_data(self) -> None:
        p = np.random.permutation(len(self.x))
        self.x = self.x[p, :]
        self.y = self.y[p, :]

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            self.shuffle_data()

        iterations = math.ceil(len(self.x) / self.batch_size)
        for i in range(0, iterations):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, self.sample_length)
            yield (
                self.x[start_index:end_index].copy(),
                self.y[start_index:end_index].copy(),
            )
