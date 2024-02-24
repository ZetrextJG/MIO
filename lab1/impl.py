from collections.abc import Callable
from typing import Any, Optional
import numpy as np


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
