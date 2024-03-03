import numpy as np
import pytest

from mygrad.preprocessors.scalers import MinMaxScaler, StandardScaler


def test_min_max_scaler():
    x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    scaler = MinMaxScaler()

    transformed_x = scaler.fit_transform(x)

    assert transformed_x.shape == (3, 3)
    predicted_x = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    assert np.allclose(transformed_x, predicted_x)


def test_min_max_no_varience():
    x = np.array([[1, 2, 3], [1, 2, 3]])
    scaler = MinMaxScaler()

    with pytest.raises(ValueError):
        scaler.fit(x)


def test_standard_scaler():
    x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    scaler = StandardScaler()

    transformed_x = scaler.fit_transform(x)
    assert np.allclose(transformed_x[1], [0, 0, 0])

    assert transformed_x.shape == (3, 3)


def test_standard_no_varience():
    x = np.array([[1, 2, 3], [1, 2, 3]])
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.fit(x)
