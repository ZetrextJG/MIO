from collections import defaultdict
from typing import Any
import numpy as np

from mygrad.components.base import Component


def indent_lines(text: str):
    return "\t" + text.replace("\n", "\n\t")


def concat_dicts(dicts: list[dict]):
    dictionary_of_list = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            dictionary_of_list[key].append(value)
    return dictionary_of_list


def prefix_dict_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {prefix + key: value for key, value in d.items()}


def sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    return dict(sorted(d.items()))


def get_parameter_number(comp: Component):
    count = 0
    for p in comp.parameters():
        count += p.data.size
    return count


def get_param_vector(comp: Component) -> np.ndarray:
    n = get_parameter_number(comp)
    params = np.ones(n, dtype=np.float64) * np.nan
    current = 0
    for p in comp.parameters():
        params[current : current + p.data.size] = p.data.flatten()
        current += p.data.size
    return params


def set_parameter_vector(comp: Component, params_vector: np.ndarray):
    current = 0
    for p in comp.parameters():
        size = p.data.size
        p.data = params_vector[current : current + size].reshape(p.data.shape)
        current += size
