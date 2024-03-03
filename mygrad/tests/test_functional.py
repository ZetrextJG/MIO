import mygrad.functional as ff
import numpy as np


def test_linear():
    x = np.array([1, 2, 3])
    A = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([1, 2])
    assert np.allclose(ff.linear(x, A, b), np.array([15, 34]))


def test_linear_batch():
    x = np.array([[1, 2, 3], [2, 4, 6]])
    A = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([1, 2])
    assert np.allclose(ff.linear(x, A, b), np.array([[15, 34], [29, 66]]))


def test_sigmoid():
    x = np.array([0, 0])
    assert np.allclose(ff.sigmoid(x), np.array([0.5, 0.5]))

    x = np.array([[0, 0]])
    assert np.allclose(ff.sigmoid(x), np.array([[0.5, 0.5]]))


def test_mse():
    x1 = np.array([1, 2, 3]).reshape(1, -1)
    x2 = np.array([1, 2, 3]).reshape(1, -1)
    assert np.allclose(ff.mse(x1, x2), np.array([0]))

    x1 = np.array([[1, 2, 3], [4, 5, 6]])
    x2 = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(ff.mse(x1, x2), np.array([[0, 0]]))
