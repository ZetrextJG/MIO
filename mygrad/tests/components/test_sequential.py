import numpy as np
import mygrad.components as mc


def test_simple_sequential_model():
    model = mc.Sequential(mc.Linear(3, 2), mc.ReLU(), mc.Linear(2, 1))
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = model.forward(x)

    grad = np.ones_like(y)
    model.backward(grad)

    assert np.any(model.components[0].parameters().__next__().grad)

    model.zero_grad()

    assert not np.any(model.components[2].parameters().__next__().grad)
