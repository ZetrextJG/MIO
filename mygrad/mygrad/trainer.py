from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np
from numpy.lib import math
from tqdm.auto import tqdm
import pandas as pd
from dataclasses import dataclass
from mygrad.components.base import Component
from mygrad.losses import Loss
from mygrad.optimizers.base import Optimizer


class Dataloader(ABC):
    batch_size: int

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        ...


class NumpyDataloader:
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int):
        assert len(x.shape) == 2, "x must be a 2D array"
        assert len(y.shape) == 2, "x must be a 2D array"
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
        assert y.shape[1] == 1, "y must be a 2D array with a single column"

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.sample_length = x.shape[0]

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(0, len(self.x), self.batch_size):
            end_index = (i + self.batch_size) % self.sample_length
            yield self.x[i:end_index].copy(), self.y[i:end_index].copy()


@dataclass
class Trainer:
    model: Component
    optimizer: Optimizer
    loss_func: Loss
    train_dataloader: Dataloader
    test_dataloader: Optional[Dataloader] = None
    validation_dataloader: Optional[Dataloader] = None
    epochs: int = 1

    def run(self):
        self.train()
        self.model.zero_grad()

        if self.test_dataloader is not None:
            self.test()
        self.model.zero_grad()

        if self.validation_dataloader is not None:
            self.validate()
        self.model.zero_grad()

    def train(self):
        print(f"### Training - {self.epochs} epochs")
        for epoch in tqdm(range(self.epochs)):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for x_batch, y_batch in tqdm(self.train_dataloader):
                y_pred = self.model.forward(x_batch)
                loss_value, loss_grad = self.loss_func.both(y_pred, y_batch)
                self.model.backward(loss_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

    def test(self):
        assert self.test_dataloader is not None, "No test dataloader provided"

        print("### Testing")
        collected_y_pred = []
        collected_y = []
        for x_batch, y_batch in tqdm(self.test_dataloader):
            y_pred = self.model.forward(x_batch)
            loss_value, loss_grad = self.loss_func.both(y_pred, y_batch)

            collected_y_pred.append(y_pred)
            collected_y.append(y_batch)

        self.optimizer.zero_grad()

        collected_y_pred = np.concatenate(collected_y_pred)
        collected_y = np.concatenate(collected_y)

        loss = self.loss_func.value(collected_y_pred, collected_y)
        print(f"Test loss: {loss}")
        return loss

    def validate(self) -> float:
        assert (
            self.validation_dataloader is not None
        ), "No validation dataloader provided"

        print("### Validation")
        collected_y_pred = []
        collected_y = []
        for x_batch, y_batch in tqdm(self.validation_dataloader):
            y_pred = self.model.forward(x_batch)
            loss_value, loss_grad = self.loss_func.both(y_pred, y_batch)

            collected_y_pred.append(y_pred)
            collected_y.append(y_batch)

        self.optimizer.zero_grad()

        collected_y_pred = np.concatenate(collected_y_pred)
        collected_y = np.concatenate(collected_y)

        loss = self.loss_func.value(collected_y_pred, collected_y)
        print(f"Validation loss: {loss}")
        return float(loss)
