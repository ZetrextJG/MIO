import numpy as np
from mygrad.layers import Linear


def test_single_value_linear():
    x = np.array([2])
    layer = Linear(1, 1)
    y = layer.forward(x)
    grad = np.ones_like(y)
    layer.backward(grad)


def test_batch_single_variable():
    x = np.array([1, 2, 3, 4]).reshape(-1, 1)
    layer = Linear(1, 1)
    y = layer.forward(x)
    grad = np.ones_like(y)
    layer.backward(grad)


def test_batch_multi_variable():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    layer = Linear(3, 2)
    y = layer.forward(x)
    grad = np.ones_like(y)
    layer.backward(grad)


def test_reset_grad():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    layer = Linear(3, 2)
    y = layer.forward(x)
    grad = np.ones_like(y)
    layer.backward(grad)
    layer.reset_grad()

    assert not np.any(layer.parameters[0].grad)
