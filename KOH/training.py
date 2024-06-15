from functools import partial
from koh import Decays, AxialCoordinateSystem, GridCoordinateSystem, Neightborhoods, euclidean_distance
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

neightborhoods_map = {
    "circle": Neightborhoods.circle,
    "gaussian": Neightborhoods.gaussian,
    "mexican_hat": Neightborhoods.mexican_hat,
}

coords_map = {
    "grid": GridCoordinateSystem,
    "hexgrid": AxialCoordinateSystem,
}

decay_map = {
    "exponential": Decays.exponential,
    "cosine": Decays.cosine,
    "full_cosine": Decays.full_cosine,
}

# HYPYERPARAMETERS:
# map_shape
# total_epochs
# start_radius
# start_learning_rate
# coords
# neightborhood
# learning_rate_decay

def train_koh(train_input: np.ndarray, hp: dict, verbose: bool = False):
    # Unpack hyperparameters
    OBSERVATION_NUM, FEATURE_DIM = train_input.shape

    MAP_SHAPE = hp["map_shape"]
    TOTAL_EPOCHS = hp["total_epochs"]
    START_RADIUS = hp["start_radius"]
    START_LEARNING_RATE = hp["start_learning_rate"]

    coords = coords_map[hp["coords"]]
    neightborhood = neightborhoods_map[hp["neightborhood"]]
    learning_rate_decay = partial(decay_map[hp["learning_rate_decay"]], total_epochs=TOTAL_EPOCHS)

    # # Model and iterations
    map_positions = coords.generate_coordinates(*MAP_SHAPE)
    map_weights = np.random.rand(*MAP_SHAPE, FEATURE_DIM)

    centroid_distances = []
    bmus_counts = []

    epoch_iter = range(TOTAL_EPOCHS)
    if verbose:
        epoch_iter = tqdm(epoch_iter)
    for current_epoch in epoch_iter:
        learning_rate = START_LEARNING_RATE * learning_rate_decay(current_epoch)

        total_centroid_distance = 0

        observation_ids = np.arange(OBSERVATION_NUM)
        np.random.shuffle(observation_ids)
        for data_vector in train_input[observation_ids]:

            # Winner (max cosine similarity)
            latent_dist = map_weights @ data_vector
            bmu = np.unravel_index(np.argmax(latent_dist), MAP_SHAPE)

            centroid_dist = euclidean_distance(map_weights[bmu], data_vector)
            total_centroid_distance += centroid_dist

            bmu_pos: np.ndarray = map_positions[bmu]
            distances = coords.distance(map_positions, bmu_pos)
            neightborhood_scale = neightborhood(
                distances, START_RADIUS, current_epoch, TOTAL_EPOCHS
            ).reshape(*MAP_SHAPE, 1)

            update = data_vector - map_weights
            map_weights += learning_rate * neightborhood_scale * update

        centroid_distances.append(total_centroid_distance)

        # After epoch eval
        all_bmus = set()
        for data_vector in train_input:
            latent_dist = map_weights @ data_vector
            bmu = np.unravel_index(np.argmax(latent_dist), MAP_SHAPE)
            all_bmus.add(bmu)

        bmus_counts.append(len(all_bmus))

    return map_weights, centroid_distances, bmus_counts, hp

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)


def calculate_accuracy(true_labels, pred_labels):
    "Calculate accuracy of the clustering using hugarian algorithm"

    cm = confusion_matrix(true_labels, pred_labels)
    indexes = np.vstack(linear_assignment(_make_cost_m(cm))).T

    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]

    acc = np.trace(cm2) / np.sum(cm2)

    return acc, cm2


def eval_koh(test_input, test_labels, map_weights):
    MAP_SHAPE = map_weights.shape[:-1]

    all_bmus = []
    labels = []
    for data_vector in test_input:
        latent_dist = map_weights @ data_vector
        bmu = np.unravel_index(np.argmax(latent_dist), MAP_SHAPE)

        if bmu not in all_bmus:
            all_bmus.append(bmu)
        labels.append(all_bmus.index(bmu))

    acc, cm = calculate_accuracy(test_labels, labels)

    return acc, cm
