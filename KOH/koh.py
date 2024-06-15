from abc import ABC
from functools import partial
import itertools
import numpy as np


def euclidean_distance(u: np.ndarray, v: np.ndarray):
    diff = u - v
    return np.sqrt(np.sum(diff**2, axis=-1))


class CoordinateSystem(ABC):
    @staticmethod
    def generate_coordinates(m: int, n: int) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def from_grid(i: int, j: int) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def distance(u: np.ndarray, v: np.ndarray):
        raise NotImplementedError


class GridCoordinateSystem(CoordinateSystem):
    @staticmethod
    def generate_coordinates(m: int, n: int) -> np.ndarray:
        return np.array(list(itertools.product(np.arange(m), np.arange(n)))).reshape(m, n, 2)

    @staticmethod
    def from_grid(i: int, j: int) -> np.ndarray:
        return np.array([i, j])

    @staticmethod
    def distance(u: np.ndarray, v: np.ndarray):
        return euclidean_distance(u, v)


class AxialCoordinateSystem(CoordinateSystem):
    @staticmethod
    def from_grid(i: int, j: int) -> np.ndarray:
        q = i - (j + (j&1)) / 2
        r = j
        return np.array([q, r])

    @staticmethod
    def generate_coordinates(m: int, n: int) -> np.ndarray:
        xs, ys = np.fromfunction(AxialCoordinateSystem.from_grid, (m, n), dtype=int)
        return np.dstack((xs, ys))

    @staticmethod
    def distance(u: np.ndarray, v: np.ndarray):
        axial_diff = u - v
        shape = axial_diff.shape
        axial_diff = axial_diff.reshape(-1, 2)
        q = axial_diff[:, 0]
        r = axial_diff[:, 1]
        distances = (np.abs(q) + np.abs(r) + np.abs(q + r)) / 2
        return distances.reshape(shape[:-1])


class Neightborhoods:
    @staticmethod
    def circle(distances: np.ndarray, radius: float = 1, time: int = 1, total_epochs: int = 1):
        radius = radius * np.exp(-time / total_epochs)
        return (distances <= radius) * 1

    @staticmethod
    def gaussian(distances: np.ndarray, radius: float = 1, time: int = 1, total_epochs: int = 1):
        t = (distances * time / radius)**2
        return np.exp(-t)

    @staticmethod
    def mexican_hat(distances: np.ndarray, radius: float = 1, time: int = 1, total_epochs: int = 1):
        t = (distances * time / radius)**2
        return (2 - 4*t)*np.exp(-t)


class Decays:
    @staticmethod
    def exponential(current_epoch: int, total_epochs: int):
        return np.exp(-current_epoch / total_epochs)

    @staticmethod
    def cosine(current_epoch: int, total_epochs: int):
        return np.cos(current_epoch / total_epochs)

    @staticmethod
    def full_cosine(current_epoch: int, total_epochs: int):
        return np.cos((np.pi/2) * current_epoch / total_epochs)
