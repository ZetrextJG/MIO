import pandas as pd

import mygrad.components as mc
import mygrad.functional as mf
from tqdm.auto import tqdm
from mygrad.losses import MeanSquareErrorLoss, CategorialCorssEntropy
from mygrad.optimizers import RMSProp
from mygrad.preprocessors import StandardScaler
from mygrad.trainers import (
    Trainer as RegressionTrainer,
    EarlyStopping,
    CategoricalClassificationTrainer,
)
from pathlib import Path
from mygrad.dataloaders import NumpyRegressionDataloader, NumpyClassificationDataloader
from mygrad.utils import sort_dict

import multiprocessing

from sklearn.model_selection import ParameterGrid

PARAMS = {
    "batch_size": [10, 20, 30, 50],
    "learning_rate": [0.01, 0.005, 0.001, 0.0005],
    "activation": [("relu", "he"), ("tanh", "xavier")],
    "dropout": [0.0, 0.15, 0.3, 0.45],
}

EPOCHS = 200
PATIENCE = 20
WARMUP = 40

REPEATS = 10

DATASETS_REGRESSION = [
    (
        "../mio1/regression/multimodal-sparse-training.csv",
        "../mio1/regression/multimodal-sparse-test.csv",
    ),
]

DATASETS_CLASSIFICATION = [
    (
        "../mio1/classification/rings5-sparse-training.csv",
        "../mio1/classification/rings5-sparse-test.csv",
    ),
    (
        "../mio1/classification/rings3-balance-training.csv",
        "../mio1/classification/rings3-balance-test.csv",
    ),
    (
        "../mio1/classification/xor3-balance-training.csv",
        "../mio1/classification/xor3-balance-test.csv",
    ),
]


train_df = pd.read_csv(DATASETS_CLASSIFICATION[0][0])
test_df = pd.read_csv(DATASETS_CLASSIFICATION[0][1])
X_train = train_df[["x", "y"]].values.reshape(-1, 2)
y_train = train_df["c"].values.reshape(-1, 1) * 1
X_test = test_df[["x", "y"]].values.reshape(-1, 2)
y_test = test_df["c"].values.reshape(-1, 1) * 1


input_normalizer = StandardScaler()

X_train_scaled = input_normalizer.fit_transform(X_train)
X_test_scaled = input_normalizer.transform(X_test)
y_train_onehot = mf.onehot_encode(y_train, 5)
y_test_onehot = mf.onehot_encode(y_test, 5)


def eval_classification_regression(params):
    model = mc.Sequential(
        mc.Linear(2, 30, init=params["activation"][1]),
        mc.ReLU() if params["activation"][0] == "relu" else mc.Tanh(),
        mc.Dropout(params["dropout"]),
        mc.Linear(30, 30, init=params["activation"][1]),
        mc.ReLU() if params["activation"][0] == "relu" else mc.Tanh(),
        mc.Dropout(params["dropout"]),
        mc.Linear(30, 30, init=params["activation"][1]),
        mc.ReLU() if params["activation"][0] == "relu" else mc.Tanh(),
        mc.Dropout(params["dropout"]),
        mc.Linear(30, 5, init="xavier"),
        mc.Softmax(),
    )
    optimizer = RMSProp(
        model.parameters(),
        learning_rate=params["learning_rate"],
        beta=0.9,  # good default value
    )
    loss = CategorialCorssEntropy()
    train_dataloader = NumpyClassificationDataloader(
        X_train_scaled,
        y_train_onehot,
        batch_size=params["batch_size"],
        shuffle=True,
        is_one_hot=True,
    )
    validation_dataloader = NumpyClassificationDataloader(
        X_test_scaled,
        y_test_onehot,
        batch_size=200,
        is_one_hot=True,
        shuffle=False,  # batch size for validation set high
    )

    trainer = CategoricalClassificationTrainer(
        model,
        optimizer,
        loss,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        plugins=[EarlyStopping(PATIENCE, warmup=WARMUP)],
    )
    outputs = trainer.train(EPOCHS)

    model.eval()
    y_test_pred = model.forward(X_test_scaled)
    model.zero_grad()

    return outputs, y_test_pred, params


def main():
    results = []
    items = list(ParameterGrid(PARAMS))

    for i in range(REPEATS):
        output_file_sgd = Path(f"./lab6/search_results_rings5_{i}.jsonl")
        with multiprocessing.Pool(6) as pool:
            for train_outputs, y_test_pred, params in tqdm(
                pool.imap_unordered(eval_classification_regression, items),
                total=len(items),
            ):
                y_pred = mf.onehot_decode(y_test_pred)
                params_sorted = sort_dict(params)
                train_outputs_sorted = sort_dict(train_outputs)
                results_dict = {
                    **params_sorted,
                    **train_outputs_sorted,
                    "fscore": float(mf.fscore(y_pred, y_test, 5)),
                }
                results.append(results_dict)
        result_df = pd.DataFrame(results)
        result_df.to_json(output_file_sgd, orient="records", lines=True)


if __name__ == "__main__":
    main()
