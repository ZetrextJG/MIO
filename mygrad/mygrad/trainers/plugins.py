from __future__ import annotations
from typing import Literal

import numpy as np
from tqdm.auto import tqdm
from mygrad.trainers.base import Plugin, Trainer
from mygrad.utils import prefix_dict_keys, sort_dict


MODES = Literal["min", "max"]


class EarlyStopping(Plugin):
    def __init__(
        self, patience: int, warmup: int = 0, metric: str = "loss", mode: MODES = "min"
    ):
        self.patience = patience
        self.warmup = warmup
        self.metric = metric
        self.mode = mode
        self.best_metric = np.inf if mode == "min" else -np.inf
        self.wait = 0

    def is_improving(self, val_metric: float) -> bool:
        if self.mode == "min":
            return val_metric < self.best_metric
        return val_metric > self.best_metric

    def on_validation_end(self, epoch: int, validation_outputs: dict):
        super().on_validation_end(epoch, validation_outputs)
        val_metric = validation_outputs[self.metric]
        if self.wait < self.warmup:
            self.best_metric = val_metric
            self.wait += 1
            return

        if self.is_improving(val_metric):
            self.best_metric = val_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience + self.warmup:
                print(f"Early stopping at epoch {epoch}")
                self.trainer.stop_training = True


class ProgressBar(Plugin):
    def __init__(self, verbose: bool = False):
        self.epoch_pbar = None
        self.verbose = verbose
        self.batch_pbar = None
        self.val_inter = None

    def on_training_start(self, epochs: int):
        super().on_training_start(epochs)
        self.epoch_pbar = tqdm(total=epochs, desc="Training epochs: ")

    def on_epoch_begin(self, epoch: int):
        super().on_epoch_begin(epoch)
        if self.verbose:
            self.batch_iter = tqdm(
                total=len(self.trainer.train_dataloader), desc="Training batches: "
            )

    def on_batch_end(self, epoch: int, batch: int, batch_outputs: dict):
        super().on_batch_end(epoch, batch, batch_outputs)
        batch_outputs = prefix_dict_keys(batch_outputs, "batch_")
        batch_outputs = sort_dict(batch_outputs)
        if self.verbose:
            self.batch_iter.update(1)
            self.batch_iter.set_postfix(batch_outputs)
            self.batch_iter.refresh()

    def on_epoch_end(self, epoch: int, epoch_outputs: dict):
        super().on_epoch_end(epoch, epoch_outputs)
        epoch_outputs = prefix_dict_keys(epoch_outputs, "train_")
        epoch_outputs = sort_dict(epoch_outputs)
        if self.verbose:
            self.batch_iter.close()
        assert self.epoch_pbar is not None
        self.epoch_pbar.update(1)
        self.epoch_pbar.set_postfix(epoch_outputs)
        self.epoch_pbar.refresh()

    def on_validation_end(self, epoch: int, val_outputs: dict):
        super().on_validation_end(epoch, val_outputs)
        val_outputs = prefix_dict_keys(val_outputs, "val_")
        val_outputs = sort_dict(val_outputs)
        assert self.epoch_pbar is not None
        self.epoch_pbar.set_postfix(val_outputs)


class ParamInfoStore(Plugin):
    def __init__(
        self,
        store_train_epochs_losses: bool = False,
        store_params_lengths: bool = False,
        store_params_angle_diffs: bool = False,
    ):
        self.store_train_epochs_losses = store_train_epochs_losses
        self.store_params_lengths = store_params_lengths
        self.store_params_angle_diffs = store_params_angle_diffs

    def init_stores(self, train_epochs: int):
        train_dataset_length = len(self.trainer.train_dataloader)
        if self.store_train_epochs_losses:
            self.train_epochs_losses = np.empty(train_epochs * train_dataset_length)
        if self.store_params_lengths:
            self.params_lengths = np.empty(train_epochs * train_dataset_length)
        if self.store_params_angle_diffs:
            self.previous_params = np.concatenate(
                [param.data.flatten() for param in self.trainer.model.parameters()]
            )
            self.params_angle_diffs = np.empty(train_epochs * train_dataset_length)

    def update_stores(self, epoch: int, batch: int, loss_value: float):
        train_dataset_length = len(self.trainer.train_dataloader)
        store_idx = epoch * train_dataset_length + batch
        if self.store_train_epochs_losses:
            self.train_epochs_losses[store_idx] = loss_value

        if self.store_params_lengths or self.store_params_angle_diffs:
            current_params = np.concatenate(
                [param.data.flatten() for param in self.trainer.model.parameters()]
            )

        if self.store_params_lengths:
            current_param_length = np.linalg.norm(current_params)  # type: ignore
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

    def on_training_start(self, epochs: int):
        super().on_training_start(epochs)
        self.init_stores(epochs)

    def on_batch_end(self, epoch: int, batch: int, batch_outputs: dict):
        super().on_batch_end(epoch, batch, batch_outputs)
        self.update_stores(epoch, batch, batch_outputs["loss"])
