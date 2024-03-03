import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray, store_grad: bool = True) -> None:
        self.data = data
        self.store_grad = store_grad
        if self.store_grad:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        if self.store_grad:
            self.grad = np.zeros_like(self.data)
