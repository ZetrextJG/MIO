import pandas as pd

import mygrad.components as mc
import mygrad.functional as mf
from tqdm.auto import tqdm
from mygrad.losses import MeanSquareErrorLoss
from mygrad.optimizers import RMSProp
from mygrad.preprocessors import StandardScaler
from mygrad.trainers import Trainer as RegressionTrainer, EarlyStopping
from pathlib import Path
from mygrad.dataloaders import NumpyRegressionDataloader
from mygrad.utils import sort_dict

import multiprocessing

from sklearn.model_selection import ParameterGrid

PARAMS = {
    "batch_size": [10, 20, 30, 50],
    "learning_rate": [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    "layers": [1, 2, 3],
    "neurons": [5, 10, 15, 20],
    "activation": [
        ("relu", "he"),
        ("tanh", "xavier"),
        ("sigmoid", "xavier"),
        ("identity", "uniform"),
    ],
}

EPOCHS = 200
PATIENCE = 10
WARMUP = 50

REPEATS = 10

train_df = pd.read_csv("../mio1/regression/multimodal-large-training.csv")
test_df = pd.read_csv("../mio1/regression/multimodal-large-test.csv")
X_train = train_df["x"].values.reshape(-1, 1)
y_train = train_df["y"].values.reshape(-1, 1)
X_test = test_df["x"].values.reshape(-1, 1)
y_test = test_df["y"].values.reshape(-1, 1)

input_normalizer = StandardScaler()
output_normalizer = StandardScaler()

X_train_scaled = input_normalizer.fit_transform(X_train)
y_train_scaled = output_normalizer.fit_transform(y_train)
X_test_scaled = input_normalizer.transform(X_test)
y_test_scaled = output_normalizer.transform(y_test)


def eval_model(params):
    model = mc.SimpleDense(
        1,
        1,
        params["layers"],
        params["neurons"],
        params["activation"][0],
        params["activation"][1],
    )
    # Output layer is linear with uniform initialization
    model = mc.Sequential(
        *model.components[:-1], mc.Linear(params["neurons"], 1, "uniform")
    )

    optimizer = RMSProp(
        model.parameters(),
        learning_rate=params["learning_rate"],
        beta=0.9,  # good default value
    )
    loss = MeanSquareErrorLoss()
    train_dataloader = NumpyRegressionDataloader(
        X_train_scaled, y_train_scaled, batch_size=params["batch_size"], shuffle=True
    )
    validation_dataloader = NumpyRegressionDataloader(
        X_test_scaled,
        y_test_scaled,
        batch_size=200,
        shuffle=False,  # batch size for validation set high
    )

    trainer = RegressionTrainer(
        model,
        optimizer,
        loss,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        plugins=[EarlyStopping(PATIENCE, warmup=WARMUP)],
    )
    outputs = trainer.train(EPOCHS)

    y_test_pred = model.forward(X_test_scaled)
    model.zero_grad()

    return outputs["loss"], output_normalizer.reverse(y_test_pred), params


def main():
    results = []
    items = list(ParameterGrid(PARAMS))
    for i in range(REPEATS):
        output_file_sgd = Path(f"./lab5/search_results_multimodal_{i}.jsonl")
        with multiprocessing.Pool() as pool:
            for train_losses, y_test_pred, params in tqdm(
                pool.imap_unordered(eval_model, items), total=len(items)
            ):
                params_sorted = sort_dict(params)
                results_dict = {
                    **params_sorted,
                    "train_losses": train_losses,
                    "mse": float(mf.mse(y_test_pred, y_test)),
                }
                results.append(results_dict)
        result_df = pd.DataFrame(results)
        result_df.to_json(output_file_sgd, orient="records", lines=True)


if __name__ == "__main__":
    main()
