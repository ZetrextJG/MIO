import numpy as np
from mygrad.models import Dense


def test_simple_dense_model():
    model = Dense(3, 1, [3, 3])
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = model.forward(x)
    assert y.shape == (3, 1)

    grad = np.ones_like(y)
    model.backward(grad)

    assert np.any(model.components[0].parameters[0].grad)

    model.reset_grad()

    assert not np.any(model.components[0].parameters[0].grad)
