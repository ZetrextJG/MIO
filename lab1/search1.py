import numpy as np
from tqdm import tqdm
import pandas as pd

from impl import MLP, Layer, Normalizer, linear, mse, sigmoid

df_train = pd.read_csv("../mio1/regression/square-simple-training.csv")

X_train = df_train["x"].values
y_train = df_train["y"].values


def build_model(b1=-1.4, b2=2, w3=15, w4=-15, b3=9):
    W1 = np.array([1, 1, 0, 0, 0]).reshape(5, 1)
    B1 = np.array([b1, b2, 0, 0, 0])

    W2 = np.array([[w3, w4, 0, 0, 0]])
    B2 = np.array([b3])

    model = MLP(
        [
            Layer(1, 5, weights=W1, bias=B1, activation_function=sigmoid),
            Layer(5, 1, weights=W2, bias=B2, activation_function=linear),
        ]
    )
    return model


if __name__ == "__main__":
    # 1 hidden 5 neurons
    x_normalizer = Normalizer()
    x = x_normalizer.fit_transform(X_train)
    # y_normalizer = Normalizer()
    # y = y_normalizer.fit_transform(y_train)

    def eval(**kwargs):
        model = build_model(**kwargs)
        y_pred = model(x)
        return mse(y_train, y_pred)

    param_grid = {
        "b1": np.linspace(-2, -1, 10),
        "b2": np.linspace(2, 3, 10),
        "w3": np.linspace(1300, 1500, 20),
        "w4": np.linspace(-1400, -1200, 20),
        "b3": np.linspace(-1000, 1000, 100),
    }

    pbar = tqdm(
        total=len(param_grid["b1"])
        * len(param_grid["b2"])
        * len(param_grid["w3"])
        * len(param_grid["w4"])
        * len(param_grid["b3"])
    )
    best_params = {}
    best_loss = float("inf")
    pbar.clear()
    for b1 in param_grid["b1"]:
        for b2 in param_grid["b2"]:
            for w3 in param_grid["w3"]:
                for w4 in param_grid["w4"]:
                    for b3 in param_grid["b3"]:
                        loss = eval(b1=b1, b2=b2, w3=w3, w4=w4, b3=b3)
                        pbar.update(1)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = {
                                "b1": b1,
                                "b2": b2,
                                "w3": w3,
                                "w4": w4,
                                "b3": b3,
                            }
    pbar.close()
    print(best_params)
    print(best_loss)
