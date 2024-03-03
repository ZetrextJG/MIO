from mygrad.layers.linear import Linear
from mygrad.models.sequential import Sequential


def Dense(input_size: int, output_size: int, hidden_neurons: list[int], **kwargs):
    sizes = [input_size] + hidden_neurons + [output_size]
    layers = [Linear(sizes[i], sizes[i + 1], **kwargs) for i in range(len(sizes) - 1)]
    return Sequential(*layers)
