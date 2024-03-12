import numpy as np
import mygrad.losses as losses


def test_dummy_loss():
    y_pred = np.array([[1, 2, 3], [2, 4, 6]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    loss = losses.DummyAbsoluteDifferenceLoss()
    assert np.allclose(loss.value(y_pred, y), np.array([2, 1, 0]))
    assert np.allclose(loss.grad(y_pred, y), np.array([[0, 0, 0], [-2, -1, 0]]))


def test_absolute_difference_loss():
    y_pred = np.array([[1, 2, 3], [2, 4, 6]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    loss = losses.AbsoluteDifferenceLoss()
    assert np.allclose(loss.value(y_pred, y), np.array([2, 1, 0]))
    assert np.allclose(loss.grad(y_pred, y), np.array([[0, 0, 0], [-1, -1, 0]]))


def test_mse_loss():
    y_pred = np.array([[1, 2, 3], [2, 4, 6]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    loss = losses.MeanSquareErrorLoss()
    assert np.allclose(loss.value(y_pred, y), np.array([2, 0.5, 0]))
    assert np.allclose(loss.grad(y_pred, y), np.array([[0, 0, 0], [-2, -1, 0]]))
