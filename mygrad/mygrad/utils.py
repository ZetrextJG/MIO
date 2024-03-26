from collections import defaultdict
from typing import Any


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
