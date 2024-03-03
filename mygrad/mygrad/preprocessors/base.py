import numpy as np
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    input_size: int
    output_size: int
    was_fit: bool

    @abstractmethod
    def fit(self, x: np.ndarray) -> np.ndarray:
        """Fit the preprocessor to the data"""
        ...

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the data using the preprocessor"""
        ...

    @abstractmethod
    def reverse(self, x: np.ndarray) -> np.ndarray:
        """Transform the data using the preprocessor"""
        ...

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit the preprocessor to the data and transform it"""
        self.fit(x)
        return self.transform(x)
