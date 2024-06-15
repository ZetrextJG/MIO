from functools import partial
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import Pool
from training import eval_koh, train_koh
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

# HYPYERPARAMETERS:
# map_shape
# total_epochs
# start_radius
# start_learning_rate
# coords
# neightborhood
# learning_rate_decay
# radius_decay

hyperparam_grid = {
    "map_shape": [(5,5)],
    "total_epochs": [30],
    "start_radius": [1, 2, 4],
    "start_learning_rate": [1e-1, 1e-2],
    "coords": ["grid", "hexgrid"],
    "neightborhood": ["gaussian", "mexican_hat"],
    "learning_rate_decay": ["exponential", "cosine", "full_cosine"],
}

uci_har_path = Path("./data/UCI HAR Dataset/UCI HAR Dataset/")
train_input = pd.read_table(uci_har_path / "train" / "X_train.txt", header=None, delim_whitespace=True).values
train_labels = pd.read_table(uci_har_path / "train" / "y_train.txt", header=None, delim_whitespace=True).values.reshape(-1)

params = list(ParameterGrid(hyperparam_grid))
training_func = partial(train_koh, train_input)
records = []

records = []
for hp in tqdm(ParameterGrid(hyperparam_grid)):
    weights, distances, bmu_counts, hp = train_koh(train_input, hp, verbose=True)

    final_cluasters = bmu_counts[-1]
    final_distances = distances[-1]

    acc, _ = eval_koh(train_input, train_labels, weights)

    records.append({
        **hp,
        "distances": distances,
        "bmu_counts": bmu_counts,
        "final_clusters": final_cluasters,
        "final_distances": final_distances,
        "accuracy": acc
    })

print(records)
output_df = pd.DataFrame(records)
output_df.to_json("./not_mnist.json", orient="records", lines=True)
