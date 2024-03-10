from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional

import numpy as np
from numpy.lib import math
from tqdm.auto import tqdm
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


class NumpyRegressionDataloader:
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

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        iterations = math.ceil(len(self.x) / self.batch_size)
        for i in range(0, iterations):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, self.sample_length)
            yield (
                self.x[start_index:end_index].copy(),
                self.y[start_index:end_index].copy(),
            )
        if self.shuffle:
            p = np.random.permutation(len(self.x))
            self.x = self.x[p, :]
            self.y = self.y[p, :]


# class AbstractTrainer(ABD):
#     @abstractmethod
#     def train(self, train_epochs: int):
#         ...
#
#     @abstractmethod
#     @overload
#     def eval(
#         self, dataloader: Dataloader, return_loss: bool
#     ) -> (np.ndarray, float):
#         ...
#
#     @abstractmethod
#     @overload
#     def eval(self, dataloader: Dataloader, return_loss: Literal[False]) -> np.ndarray:
#         ...`)


@dataclass
class RegressionTrainer:
    model: Component
    optimizer: Optimizer
    loss_func: Loss
    train_dataloader: Dataloader
    validation_dataloader: Optional[Dataloader] = None
    store_train_epochs_losses: bool = False
    store_params_lengths: bool = False
    store_params_angle_diffs: bool = False

    def init_stores(self, train_epochs: int):
        train_dataset_length = len(self.train_dataloader)
        if self.store_train_epochs_losses:
            self.train_epochs_losses = np.empty(train_epochs * train_dataset_length)

        if self.store_params_lengths:
            self.params_lengths = np.empty(train_epochs * train_dataset_length)
        if self.store_params_angle_diffs:
            self.previous_params = np.concatenate(
                [param.data.flatten() for param in self.model.parameters()]
            )
            self.params_angle_diffs = np.empty(train_epochs * train_dataset_length)

    def update_stores(self, epoch: int, batch: int, loss_value: float):
        train_dataset_length = len(self.train_dataloader)
        store_idx = epoch * train_dataset_length + batch
        if self.store_train_epochs_losses:
            self.train_epochs_losses[store_idx] = loss_value

        if self.store_params_lengths or self.store_params_angle_diffs:
            current_params = np.concatenate(
                [param.data.flatten() for param in self.model.parameters()]
            )

        if self.store_params_lengths:
            current_param_length = np.linalg.norm(current_params)
            self.params_lengths[store_idx] = current_param_length

        if self.store_params_angle_diffs:
            a_dot_b = np.dot(current_params, self.previous_params)  # type: ignore
            if self.store_params_lengths:
                norm_a = current_param_length  # type: ignore
            else:
                norm_a = np.linalg.norm(current_params)  # type: ignore
            norm_b = np.linalg.norm(self.previous_params)

            angle_diff = np.arccos(a_dot_b / (norm_a * norm_b))
            self.params_angle_diffs[store_idx] = float(angle_diff)

    def train(
        self, train_epochs: int, verbose: bool = False
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns a array of losses after each epoch evaluated on training dataset and optionally on the validation dataset"""
        self.init_stores(train_epochs)

        train_losses = np.empty(train_epochs)
        if self.validation_dataloader is not None:
            validation_losses = np.empty(train_epochs)
        else:
            validation_losses = None

        print(f"### Training - {train_epochs} epochs")
        for epoch in tqdm(range(train_epochs)):
            iterator = enumerate(self.train_dataloader)
            if verbose:
                iterator = tqdm(iterator, total=len(self.train_dataloader))

            for batch_id, (x_batch, y_batch) in iterator:
                y_pred = self.model.forward(x_batch)
                loss_value, loss_grad = self.loss_func.both(y_pred, y_batch)

                if float(loss_value) > 1e12:
                    raise ValueError(f"Loss value exploded: {loss_value}")

                self.model.backward(loss_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.update_stores(epoch, batch_id, float(loss_value))

            self.optimizer.zero_grad()
            self.optimizer.end_epoch()

            train_losses[epoch] = self.eval(self.train_dataloader)[1]
            if self.validation_dataloader is not None:
                assert validation_losses is not None
                validation_losses[epoch] = self.eval(self.validation_dataloader)[1]
            self.optimizer.zero_grad()

        return train_losses, validation_losses

    def eval(self, dataloader: Dataloader) -> tuple[np.ndarray, float]:
        all_ys = []
        all_ys_pred = []
        for x_batch, y_batch in dataloader:
            y_pred = self.model.forward(x_batch)
            all_ys.append(y_batch)
            all_ys_pred.append(y_pred)
        # self.model.zero_grad()

        all_ys = np.concatenate(all_ys, axis=0)
        all_ys_pred = np.concatenate(all_ys_pred, axis=0)
        return all_ys, float(self.loss_func.value(all_ys_pred, all_ys))

    # def test(self):
    #     assert self.test_dataloader is not None, "No test dataloader provided"
    #
    #     print("### Testing")
    #     collected_y_pred = []
    #     collected_y = []
    #     for x_batch, y_batch in tqdm(self.test_dataloader):
    #         y_pred = self.model.forward(x_batch)
    #         loss_value, loss_grad = self.loss_func.both(y_pred, y_batch)
    #
    #         collected_y_pred.append(y_pred)
    #         collected_y.append(y_batch)
    #
    #     self.optimizer.zero_grad()
    #
    #     collected_y_pred = np.concatenate(collected_y_pred)
    #     collected_y = np.concatenate(collected_y)
    #
    #     loss = self.loss_func.value(collected_y_pred, collected_y)
    #     print(f"Test loss: {loss}")
    #     return loss
    #
    # def validate(self) -> float:
    #     assert (
    #         self.validation_dataloader is not None
    #     ), "No validation dataloader provided"
    #
    #     print("### Validation")
    #     collected_y_pred = []
    #     collected_y = []
    #     for x_batch, y_batch in tqdm(self.validation_dataloader):
    #         y_pred = self.model.forward(x_batch)
    #         loss_value, loss_grad = self.loss_func.both(y_pred, y_batch)
    #
    #         collected_y_pred.append(y_pred)
    #         collected_y.append(y_batch)
    #
    #     self.optimizer.zero_grad()
    #
    #     collected_y_pred = np.concatenate(collected_y_pred)
    #     collected_y = np.concatenate(collected_y)
    #
    #     loss = self.loss_func.value(collected_y_pred, collected_y)
    #     print(f"Validation loss: {loss}")
    #     return float(loss)
