import pandas as pd
import numpy as np

import mygrad.components as mc
import mygrad.functional as mf
from tqdm.auto import tqdm
from mygrad.losses import MeanSquareErrorLoss
from mygrad.optimizers import SGD, RMSProp
from mygrad.preprocessors import MinMaxScaler, StandardScaler, IdentityScaler
from mygrad.trainers import Trainer as RegressionTrainer, ProgressBar
from pathlib import Path
from mygrad.dataloaders import NumpyRegressionDataloader

from sklearn.model_selection import ParameterGrid

# Training df, test df, target MSE
DATASETS = [
    (
        "../../mio1/regression/square-large-training.csv",
        "../../mio1/regression/square-large-test.csv",
    ),
    (
        "../../mio1/regression/steps-large-training.csv",
        "../../mio1/regression/steps-large-test.csv",
    ),
    (
        "../../mio1/regression/multimodal-large-training.csv",
        "../../mio1/regression/multimodal-large-test.csv",
    ),
]

SDG_PARAMS = {
    "batch_size": [10, 20, 30, 50],
    "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
    "beta": [0.5, 0.8, 0.9],
    "layers": [1, 2, 3, 4],
    "neurons": [10, 15, 30],
    "activation": [("relu", "he"), ("tanh", "xavier")],
}

EPOCHS = 20


def eval_sdg_model(X_train_scaled, y_train_scaled, X_test_scaled, **kwargs):
    model = mc.SimpleDense(
        1,
        1,
        kwargs["layers"],
        kwargs["neurons"],
        kwargs["activation"][0],
        kwargs["activation"][1],
    )
    model = mc.Sequential(
        *model.components[:-1], mc.Linear(kwargs["neurons"], 1, "uniform")
    )

    optimizer = SGD(
        model.parameters(),
        learning_rate=kwargs["learning_rate"],
        momentum=kwargs["beta"],
        dampening=kwargs["beta"],
    )
    loss = MeanSquareErrorLoss()
    train_dataloader = NumpyRegressionDataloader(
        X_train_scaled, y_train_scaled, batch_size=kwargs["batch_size"], shuffle=True
    )
    trainer = RegressionTrainer(model, optimizer, loss, train_dataloader)
    train_losses = trainer.train(EPOCHS)["loss"]

    y_test_pred = model.forward(X_test_scaled)
    model.zero_grad()

    return train_losses, y_test_pred


def main():
    for train_file, test_file in DATASETS:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        X_train = train_df["x"].values.reshape(-1, 1)
        y_train = train_df["y"].values.reshape(-1, 1)
        X_test = test_df["x"].values.reshape(-1, 1)
        y_test = test_df["y"].values.reshape(-1, 1)

        input_normalizer = StandardScaler()
        output_normalizer = StandardScaler()

        X_train_scaled = input_normalizer.fit_transform(X_train)
        y_train_scaled = output_normalizer.fit_transform(y_train)
        X_test_scaled = input_normalizer.transform(X_test)

        results = []
        output_file_sgd = Path(f"search_results_sgd_{train_file.split('/')[-1]}.jsonl")
        for params in tqdm(ParameterGrid(SDG_PARAMS)):
            train_losses, y_test_pred = eval_sdg_model(
                X_train_scaled, y_train_scaled, X_test_scaled, **params
            )
            y_test_pred = output_normalizer.reverse(y_test_pred)
            mse = mf.mse(y_test_pred, y_test)
            results.append(
                {
                    "params": params,
                    "train_losses": train_losses,
                    "mse": mse,
                }
            )
        result_df = pd.DataFrame(results)
        result_df.to_json(output_file_sgd, orient="records", lines=True)


if __name__ == "__main__":
    main()
