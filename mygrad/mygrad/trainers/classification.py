from typing_extensions import override
from mygrad.dataloaders import Dataloader
from mygrad.trainers.base import Trainer
from dataclasses import dataclass
import numpy as np
from mygrad.functional import fscore, onehot_decode


@dataclass
class BinaryClassificationTrainer(Trainer):
    @override
    def eval(self, dataloader: Dataloader) -> dict:
        self.model.eval()
        all_ys = []
        all_ys_pred = []
        for x_batch, y_batch in dataloader:
            y_pred = self.model.forward(x_batch)
            all_ys.append(y_batch)
            all_ys_pred.append(y_pred)
        self.model.zero_grad()
        self.model.train()

        all_ys = np.concatenate(all_ys, axis=0).astype(int)
        all_ys_pred = np.concatenate(all_ys_pred, axis=0)
        loss = self.loss_func.value(all_ys_pred, all_ys)

        all_ys_pred = (all_ys_pred > 0.5).astype(int)
        f_score = fscore(all_ys, all_ys_pred, 2)

        return {"loss": float(loss), "fscore": float(f_score)}


@dataclass
class CategoricalClassificationTrainer(Trainer):
    @override
    def eval(self, dataloader: Dataloader) -> dict:
        self.model.eval()
        all_ys = []
        all_ys_pred = []
        for x_batch, y_batch in dataloader:
            y_pred = self.model.forward(x_batch)
            all_ys.append(y_batch)
            all_ys_pred.append(y_pred)
        self.model.zero_grad()
        self.model.train()

        all_ys = np.concatenate(all_ys, axis=0).astype(int)
        all_ys_pred = np.concatenate(all_ys_pred, axis=0)
        loss = self.loss_func.value(all_ys_pred, all_ys)

        all_ys_pred = np.argmax(
            all_ys_pred, axis=1, keepdims=True
        )  # This is not one hot encoded
        all_ys = onehot_decode(all_ys)
        f_score = fscore(all_ys, all_ys_pred, y_batch.shape[1])

        return {"loss": float(loss), "fscore": float(f_score)}
