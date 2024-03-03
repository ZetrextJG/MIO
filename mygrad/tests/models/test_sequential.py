import numpy as np
from mygrad.layers import Linear
from mygrad.models import Sequential
from mygrad.models import Dense


def test_simple_sequential_model():
    model = Sequential(Linear(3, 2), Linear(2, 1))
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = model.forward(x)

    grad = np.ones_like(y)
    model.backward(grad)

    assert np.any(model.components[0].parameters().__next__().grad)

    model.reset_grad()

    assert not np.any(model.components[1].parameters().__next__().grad)


def test_sequential_with_dense():
    submodel1 = Dense(3, 2, [3, 3])
    submodel2 = Dense(2, 1, [2, 2])
    model = Sequential(submodel1, submodel2)

    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    model.forward(x)
