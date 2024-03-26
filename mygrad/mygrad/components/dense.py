import numpy as np
from mygrad.components.sequential import Sequential
from mygrad.components.linear import Linear, INIT_METHODS_STR
from mygrad.components.activations import ACTIVATION_METHODS, ACTIVATION_METHODS_STR


def Dense(
    neurons: list[int],
    activation: ACTIVATION_METHODS_STR = "relu",
    init: INIT_METHODS_STR = "xavier",
) -> Sequential:
    """Returns a sequential model of linear layers with activation layers in between.

    Example:
    Dense([784, 100, 10], "relu", "xavier") == Sequential(
        Linear(784, 100, "xavier"),
        ReLU(),
        Linear(100, 10, "xavier")
    )
    """
    assert len(neurons) > 1, "At least two sets of neurons are required"
    Activation = ACTIVATION_METHODS[activation]
    layers = []
    for i in range(len(neurons) - 1):
        layers.append(Linear(neurons[i], neurons[i + 1], init))
        if i != len(neurons) - 2:
            layers.append(Activation())
    return Sequential(*layers)


def SimpleDense(
    input_size: int,
    output_size: int,
    hidden_layers: int,
    neurons_per_layer: int,
    activation: ACTIVATION_METHODS_STR = "relu",
    init: INIT_METHODS_STR = "xavier",
):
    """Returns a sequential model of linear layers with activation layers in between."""
    layers_list: list[int] = [neurons_per_layer] * hidden_layers
    layers_list.insert(0, input_size)
    layers_list.append(output_size)
    return Dense(layers_list, activation, init)
