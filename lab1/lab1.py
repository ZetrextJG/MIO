from collections.abc import Callable
from typing import Any, Optional
from matplotlib.pyplot import xkcd
import numpy as np
from tqdm import tqdm
import pandas as pd

# Square simple

df_train = pd.read_csv("./mio1/regression/square-simple-training.csv")
df_test = pd.read_csv("./mio1/regression/square-simple-test.csv")

X_train = df_train["x"].values
y_train = df_train["y"].values

X_test = df_test["x"].values
y_test = df_test["y"].values


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def linear(x: np.ndarray) -> np.ndarray:
    return x


class Normalizer:
    def __init__(self) -> None:
        self.mean = 0
        self.std = 1

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_function: Callable[[np.ndarray], np.ndarray] = sigmoid,
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ) -> None:
        # Initialize weights and bias
        self.input_size = input_size
        self.output_size = output_size
        if weights is not None:
            assert weights.shape == (output_size, input_size)
            self.weights = weights
        else:
            self.weights = np.random.normal(0, 1, (output_size, input_size))

        if bias is not None:
            assert bias.shape == (output_size,)
            self.bias = bias
        else:
            self.bias = np.zeros(output_size)

        self._activation_function = activation_function

    def forward(self, input: np.ndarray) -> np.ndarray:
        activation_input = input @ self.weights.T + self.bias
        return self._activation_function(activation_input)


class MLP:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        self.validate_layers()
        self.input_size = layers[0].input_size
        self.output_size = layers[-1].output_size

    def validate_layers(self):
        self.last_layer_output = -1
        for layer in self.layers:
            if self.last_layer_output != -1:
                assert self.last_layer_output == layer.input_size
                self.last_layer_output = layer.output_size

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert len(input.shape) in [1, 2], "Input shape must be 1D or 2D"
        original_shape = input.shape
        if len(input.shape) == 1:
            input = input.reshape(-1, 1)
        else:
            assert input.shape[1] == self.input_size

        for i, layer in enumerate(self.layers):
            input = layer.forward(input)

        return input.reshape(original_shape)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)

    def forward_loss(self, input: np.ndarray, y_true: np.ndarray) -> np.floating[Any]:
        y_pred = self.forward(input)
        return mse(y_true, y_pred)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating[Any]:
    return np.mean((y_true - y_pred) ** 2)


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

    # pbar = tqdm(
    #     total=len(param_grid["b1"])
    #     * len(param_grid["b2"])
    #     * len(param_grid["w3"])
    #     * len(param_grid["w4"])
    #     * len(param_grid["b3"])
    # )
    # best_params = {}
    # best_loss = float("inf")
    # pbar.clear()
    # for b1 in param_grid["b1"]:
    #     for b2 in param_grid["b2"]:
    #         for w3 in param_grid["w3"]:
    #             for w4 in param_grid["w4"]:
    #                 for b3 in param_grid["b3"]:
    #                     loss = eval(b1=b1, b2=b2, w3=w3, w4=w4, b3=b3)
    #                     pbar.update(1)
    #                     if loss < best_loss:
    #                         best_loss = loss
    #                         best_params = {
    #                             "b1": b1,
    #                             "b2": b2,
    #                             "w3": w3,
    #                             "w4": w4,
    #                             "b3": b3,
    #                         }
    # pbar.close()
    # print(best_params)
    # print(best_loss)
    #
    params = {
        "b1": -2.0,
        "b2": 2.5555555555555554,
        "w3": 1415.7894736842106,
        "w4": -1305.2631578947369,
        "b3": 919.1919191919192,
    }
    model = build_model(**params)
    print(model.forward_loss(x, y_train))
    test = x_normalizer.transform(X_test)
    print(model.forward_loss(test, y_test))
