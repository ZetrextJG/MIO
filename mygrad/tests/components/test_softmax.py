import math
import numpy as np
from mygrad.components import Softmax


def test_single_softmax():
    x = np.array([[0, math.log(3)]])
    sm = Softmax()

    s = sm.forward(x)
    assert np.allclose(s, np.array([[0.25, 0.75]]))

    grad = np.array([[-1, 1]])
    back = sm.backward(grad)
    expexted = np.array([[-0.375, 0.375]])

    assert np.allclose(back, expexted)


def test_multiple_softmax():
    x = np.array([[0, math.log(3)], [1, 1]])
    sm = Softmax()

    s = sm.forward(x)
    assert np.allclose(s, np.array([[0.25, 0.75], [0.5, 0.5]]))

    grad = np.array([[-1, 1], [1, -1]])
    expexted = np.array([[-0.375, 0.375], [0.5, -0.5]])
    back = sm.backward(grad)

    assert np.allclose(back, expexted)
