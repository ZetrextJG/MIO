import numpy as np
import mygrad.components as mc


def test_simple_dense_model():
    model = mc.Dense([3, 2, 1], "relu", "xavier")
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = model.forward(x)

    grad = np.ones_like(y)
    model.backward(grad)

    model.zero_grad()
    assert not np.any(model.components[2].parameters().__next__().grad)
