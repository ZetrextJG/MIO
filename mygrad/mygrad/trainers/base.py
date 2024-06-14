from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional, TypedDict

import numpy as np
from dataclasses import dataclass, field
from mygrad.components.base import Component
from mygrad.dataloaders import Dataloader
from mygrad.losses import Loss
from mygrad.optimizers.base import Optimizer
from mygrad.utils import concat_dicts


class Plugin:
    trainer: Trainer

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_training_start(self, epochs: int):
        assert self.trainer is not None

    def on_training_end(self):
        assert self.trainer is not None

    def on_epoch_begin(self, epoch: int):
        assert self.trainer is not None

    def on_epoch_end(self, epoch: int, epoch_outputs: dict):
        assert self.trainer is not None

    def on_batch_begin(self, epoch: int, batch: int):
        assert self.trainer is not None

    def on_batch_end(self, epoch: int, batch: int, batch_outputs: dict):
        assert self.trainer is not None

    def on_validation_begin(self, epoch: int):
        assert self.trainer is not None

    def on_validation_end(self, epoch: int, validation_outputs: dict):
        assert self.trainer is not None


@dataclass
class Trainer:
    model: Component
    optimizer: Optimizer
    loss_func: Loss
    train_dataloader: Dataloader
    validation_dataloader: Optional[Dataloader] = None
    stop_training: bool = False
    plugins: list[Plugin] = field(default_factory=list)

    def __post_init__(self):
        for plugin in self.plugins:
            plugin.set_trainer(self)

    def start_training(self, train_epochs: int):
        self.model.train()
        for plugin in self.plugins:
            plugin.on_training_start(train_epochs)

    def start_epoch(self, epoch: int):
        self.model.train()
        for plugin in self.plugins:
            plugin.on_epoch_begin(epoch)

    def end_epoch(self, epoch: int, epoch_outputs: dict):
        for plugin in self.plugins:
            plugin.on_epoch_end(epoch, epoch_outputs)
        self.model.eval()

    def start_batch(self, epoch: int, batch: int):
        for plugin in self.plugins:
            plugin.on_batch_begin(epoch, batch)

    def end_batch(self, epoch: int, batch: int, batch_outputs: dict):
        for plugin in self.plugins:
            plugin.on_batch_end(epoch, batch, batch_outputs)

    def start_validation(self, epoch: int):
        self.model.eval()
        for plugin in self.plugins:
            plugin.on_validation_begin(epoch)

    def end_validation(self, epoch: int, val_outputs: dict):
        for plugin in self.plugins:
            plugin.on_validation_end(epoch, val_outputs)
        self.model.train()

    def train(self, train_epochs: int) -> dict:
        self.start_training(train_epochs)
        results = []
        for epoch in range(train_epochs):
            result = self.train_epoch(epoch)
            if self.stop_training:
                break
            if self.validation_dataloader is not None:
                val_result = self.validation_epoch(epoch)
                for key, value in val_result.items():
                    result[f"val_{key}"] = value
            results.append(result)
        return concat_dicts(results)

    def eval(self, dataloader: Dataloader) -> dict:
        """Overwrite if needed"""
        self.model.eval()
        all_ys = []
        all_ys_pred = []
        for x_batch, y_batch in dataloader:
            y_pred = self.model.forward(x_batch)
            all_ys.append(y_batch)
            all_ys_pred.append(y_pred)
        self.model.zero_grad()
        self.model.train()

        all_ys = np.concatenate(all_ys, axis=0)
        all_ys_pred = np.concatenate(all_ys_pred, axis=0)
        loss = self.loss_func.value(all_ys_pred, all_ys)
        return {"loss": float(loss)}

    def train_epoch(self, epoch: int) -> dict:
        """Overwrite if needed"""
        self.start_epoch(epoch)
        for batch_id, (x_batch, y_batch) in enumerate(self.train_dataloader):
            self.start_batch(epoch, batch_id)

            # Main training
            y_pred = self.model.forward(x_batch)
            loss_value, loss_grad = self.loss_func.both(y_pred, y_batch)

            if float(loss_value) > 1e12:
                raise ValueError(f"Loss value exploded: {loss_value}")

            self.model.backward(loss_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.end_batch(epoch, batch_id, {"loss": float(loss_value)})

            if self.stop_training:
                break

        eval_outputs = self.eval(self.train_dataloader)
        self.optimizer.end_epoch()
        self.end_epoch(epoch, eval_outputs)
        return eval_outputs

    def validation_epoch(self, epoch: int) -> dict:
        """Overwrite if needed"""
        assert self.validation_dataloader is not None
        self.start_validation(epoch)
        outputs = self.eval(self.validation_dataloader)
        self.end_validation(epoch, outputs)
        return outputs
