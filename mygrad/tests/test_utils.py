import mygrad.components as mc
import numpy as np
from mygrad.utils import get_param_vector, get_parameter_number, set_parameter_vector


def test_getting_param_count():
    network = mc.Sequential(
        mc.Linear(2, 3), mc.ReLU(), mc.Linear(3, 3), mc.ReLU(), mc.Linear(3, 1)
    )
    true_param_count = 6 + 3 + 9 + 3 + 3 + 1
    param_count = get_parameter_number(network)
    assert true_param_count == param_count


def test_getting_param_vector():
    network = mc.Sequential(
        mc.Linear(2, 3), mc.ReLU(), mc.Linear(3, 3), mc.ReLU(), mc.Linear(3, 1)
    )
    param_vector = get_param_vector(network)
    assert not np.isnan(param_vector).any()


def test_mutate_param_vector():
    network = mc.Sequential(
        mc.Linear(2, 3), mc.ReLU(), mc.Linear(3, 3), mc.ReLU(), mc.Linear(3, 1)
    )
    param_vector = get_param_vector(network)
    param_vector[0] = 100
    set_parameter_vector(network, param_vector)

    new_param_vector = get_param_vector(network)
    assert new_param_vector[0] == 100
    assert np.allclose(new_param_vector, param_vector)
