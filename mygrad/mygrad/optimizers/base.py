from abc import ABC, abstractmethod

from mygrad.parameters import Parameter


class Optimizer(ABC):
    params: list[Parameter]

    @abstractmethod
    def step(self):
        """Performs a single optimization step"""
        ...

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def end_epoch(self):
        """Called at the end of each epoch"""
        pass
