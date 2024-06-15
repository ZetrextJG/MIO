from functools import partial
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import Pool
# from training import eval_koh, train_koh
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
    "map_shape": [(3,3), (4,4), (5,5), (10,10)],
    "total_epochs": [50],
    "start_radius": [1, 2, 4],
    "start_learning_rate": [1e-1, 5e-2, 1e-2],
    "coords": ["grid", "hexgrid"],
    "neightborhood": ["circle", "gaussian", "mexican_hat"],
    "learning_rate_decay": ["exponential", "cosine", "full_cosine"],
}

df = pd.read_csv("./data/cube.csv")
train_df = df[["x", "y", "z"]]
train_df.head()
train_input = train_df.values
train_labels = df["c"].values


params = list(reversed(ParameterGrid(hyperparam_grid)))
training_func = partial(train_koh, train_input)
records = []

with Pool(6) as p:
    iterator = tqdm(p.imap_unordered(training_func, params), total=len(params))
    for (weights, distances, bmu_counts, hp) in iterator:

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

# Save records
output_df = pd.DataFrame(records)
output_df.to_json("./cube.json", orient="records", lines=True)
