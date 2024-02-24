import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from impl import MLP, Layer, linear, mse, sigmoid

df_train = pd.read_csv("../mio1/regression/steps-large-training.csv")

X_train = df_train["x"].values
y_train = df_train["y"].values

# plt.plot(X_train, y_train, "go")
# plt.show()
# sys.exit(0)


def build_model(b1=0, b2=0, b3=0, w1=1, w2=1, w3=1, b4=0):
    W1 = np.array([1000, 1000, 1000, 0, 0]).reshape(5, 1)
    B1 = np.array([b1, b2, b3, 0, 0])

    W2 = np.array([[w1, w2, w3, 0, 0]])
    B2 = np.array([b4])

    model = MLP(
        [
            Layer(1, 5, weights=W1, bias=B1, activation_function=sigmoid),
            Layer(5, 1, weights=W2, bias=B2, activation_function=linear),
        ]
    )
    return model


if __name__ == "__main__":
    # 1 hidden 5 neurons
    def eval(**kwargs):
        model = build_model(**kwargs)
        y_pred = model(X_train)
        return mse(y_train, y_pred)

    grid = {
        "b1": np.linspace(-505, -495, 10),
        "b2": np.linspace(-1505, -1495, 10),
        "b3": np.linspace(495, 505, 10),
        "w1": np.linspace(75, 85, 10),
        "w2": np.linspace(75, 85, 10),
        "w3": np.linspace(75, 85, 10),
        "b4": [y_train.min()],
    }

    best_params = {}
    best_loss = float("inf")
    params_grid = ParameterGrid(grid)
    for params in tqdm(params_grid):
        loss = eval(**params)
        if loss < best_loss:
            best_loss = loss
            best_params = params

    print(best_params)
    print(best_loss)
