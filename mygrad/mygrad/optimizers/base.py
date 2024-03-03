import numpy as np
from abc import ABC, abstractmethod

from mygrad.models import Model


def Optimizer(ABC):
    parameters: tuple[Para]

    @abstractmethod
    def step(self):
        """Performs a single optimization step"""
        ...
