
from functools import partial
from koh import Decays, AxialCoordinateSystem, GridCoordinateSystem, Neightborhoods, euclidean_distance
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import imageio
import numpy as np
import pandas as pd
import seaborn as sns

# RUN

cube_df = pd.read_csv("./data/hexagon.csv")
train_df = cube_df[["x", "y"]]
train_df.head()
train_input = train_df.values

# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.show()


OBSERVATION_NUM, FEATURE_DIM = train_input.shape

MAP_M = 3
MAP_N = 3
MAP_SHAPE = (MAP_M, MAP_N)
TOTAL_EPOCHS = 50

START_RADIUS = 2
START_LEARNING_RATE = 1e-2

coords = AxialCoordinateSystem
neightborhood = Neightborhoods.circle
learning_rate_decay = partial(Decays.cosine, total_epochs=TOTAL_EPOCHS)


# # Model and iterations
map_positions = coords.generate_coordinates(*MAP_SHAPE)
map_weights = np.random.rand(*MAP_SHAPE, FEATURE_DIM)

print(map_positions.shape)

images_dir = Path("./images")
images_dir.mkdir(exist_ok=True)

frame_paths = []
def plot_map_weights(map_weights, time):
    plt.scatter(train_input[:, 0], train_input[:, 1], marker="o")
    plt.scatter(map_weights[:, :, 0], map_weights[:, :, 1], marker="x", color="red", s=150)
    output_file = images_dir / f"time_{time}.png"
    plt.savefig(output_file)
    frame_paths.append(output_file)
    plt.close()

plot_map_weights(map_weights, 0)

# Training

centroid_distances = []
bmus_counts = []

for current_epoch in tqdm(range(TOTAL_EPOCHS)):
    learning_rate = START_LEARNING_RATE * learning_rate_decay(current_epoch)

    observation_ids = np.arange(OBSERVATION_NUM)
    np.random.shuffle(observation_ids)

    total_centroid_distance = 0
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
    plot_map_weights(map_weights, current_epoch + 1)

    # Eval
    all_bmus = []
    labels = []
    for data_vector in train_input:
        latent_dist = map_weights @ data_vector
        bmu = np.unravel_index(np.argmax(latent_dist), MAP_SHAPE)

        if bmu not in all_bmus:
            all_bmus.append(bmu)
        labels.append(all_bmus.index(bmu))

    bmus_counts.append(len(all_bmus))


from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

true_labels = cube_df["c"].values

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)


def calculate_accuracy(true_lables, pred_labels):
    "Calculate accuracy of the clustering using hugarian algorithm"

    cm = confusion_matrix(true_labels, pred_labels)
    indexes = np.vstack(linear_assignment(_make_cost_m(cm))).T

    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]

    acc = np.trace(cm2) / np.sum(cm2)

    return acc, cm2





plt.xlabel("Epoka")
plt.ylabel("Suma odległości danych od ich BMU")
plt.plot(centroid_distances)
plt.show()

plt.xlabel("Epoka")
plt.ylabel("Liczba BMU")
plt.scatter(range(TOTAL_EPOCHS), bmus_counts)
plt.show()


frames = []
for frame_path in frame_paths:
    image = imageio.v2.imread(frame_path)
    frames.append(image)

imageio.mimsave(
    './learning.gif',
    frames,
    fps = 10
)
