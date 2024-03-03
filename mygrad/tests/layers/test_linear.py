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


def test_zero_grad():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    layer = Linear(3, 2)
    y = layer.forward(x)
    grad = np.ones_like(y)
    layer.backward(grad)
    layer.zero_grad()

    assert not np.any(layer.parameters().__next__().grad)


def test_parameters():
    layer = Linear(3, 2)
    params = list(layer.parameters())
    assert len(params) == 2
    layer.zero_grad()

    first_param = params[0]
    assert np.allclose(first_param.grad, np.zeros_like(first_param.grad))
