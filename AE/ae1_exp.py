from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from functools import partial
import numpy as np
import pandas as pd
import math


class Mutation:
    @staticmethod
    def gaussian(X: np.ndarray, sigma: float = 0.1, p: float = 0.2) -> np.ndarray:
        """
        Apply gaussian mutation to the input matrix X with a given probability p
        """
        mask = (np.random.random(X.shape) < p) * 1
        noise = np.random.normal(0, sigma, X.shape)
        return X + mask * noise

    @staticmethod
    def flip(X: np.ndarray, p: float = 0.2) -> np.ndarray:
        """
        Apply flip mutation to the input matrix X with a given probability p
        """
        mask = (np.random.random(X.shape) < p) * 1
        return (1 - mask) * X + mask * (1 - X)


class Crossover:

    @staticmethod
    def get_parents(population: np.ndarray, crossover_p: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the population into two groups of parents
        """
        prob_vector = np.random.random(population.shape[0])
        reproductive_population = population[prob_vector < crossover_p]
        idx = np.arange(reproductive_population.shape[0])
        np.random.shuffle(idx)
        reproductive_population = reproductive_population[idx]

        split_idx = reproductive_population.shape[0] // 2

        return reproductive_population[:split_idx], reproductive_population[split_idx:(2*split_idx)]

    @staticmethod
    def pointwise(parents1: np.ndarray, parents2: np.ndarray, split_portion: float = 0.5) -> np.ndarray:
        """
        Apply pointwise crossover to the input matrices parent1 and parent2 with a given probability p
        """
        assert parents1.shape == parents2.shape
        split_point = int(parents1.shape[1] * split_portion)
        return np.concatenate([parents1[:, :split_point], parents2[:, split_point:]], axis=1)

    @staticmethod
    def mean(parents1: np.ndarray, parents2: np.ndarray) -> np.ndarray:
        """
        Apply mean crossover to the input matrices parent1 and parent2 with a given probability p
        """
        assert parents1.shape == parents2.shape
        return (parents1 + parents2) / 2

        
def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalize the input matrix X
    """
    return (X - np.min(X)) / (np.max(X) - np.min(X))


class Selection:

    @staticmethod
    def roulette(population: np.ndarray, fitness: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Apply roulette selection to the input population. Select n individuals based on their fitness
        """
        fitness = normalize(fitness)
        total_fitness = np.sum(fitness)
        probabilities = fitness / total_fitness
        idx = np.random.choice(population.shape[0], n, p=probabilities)
        return population[idx]


    @staticmethod
    def elitist(population: np.ndarray, fitness: np.ndarray, n: int = 1, elite_percentage: float = 0.15) -> np.ndarray:
        """
        Apply elitist selection to the input population. Select the n fittest individuals
        """
        fitness = normalize(fitness)
        m = int(population.shape[0] * elite_percentage)
        sorted_population = population[np.argsort(fitness)[::-1]]

        population = list(sorted_population[:m])
        left_over_population = list(sorted_population[m:])
        while len(population) < n:
            probs = np.array([fitness[i] for i in range(len(left_over_population))])
            probs = probs / probs.sum()
            idx = np.random.choice(len(left_over_population), p=probs)
            ind = left_over_population.pop(idx)
            population.append(ind)

        return np.array(population)


    @staticmethod
    def fittest(population: np.ndarray, fitness: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Apply fittest selection to the input population. Select the n fittest individuals
        """
        idx = np.argsort(fitness)[::-1][:n]
        return population[idx]

CROSSOVERS = {
    "pointwise": Crossover.pointwise,
    "mean": Crossover.mean
}

SELECTIONS = {
    "roulette": Selection.roulette,
    "elitist15": partial(Selection.elitist, elite_percentage=0.15),
    "elitist30": partial(Selection.elitist, elitism_selection=0.3),
    "fittest": Selection.fittest
}


def simple_parabolid3D(X):
    return X[:, 0] ** 2 + X[:, 1] ** 2 + 2 * (X[:, 2] ** 2)


def rastrigin10D(X):
    A = 10
    return A * X.shape[1] + np.sum(X ** 2 - A * np.cos(2 * math.pi * X), axis=1)


FUNCTIONS = {
    "parabolid3D": lambda x: -simple_parabolid3D(x),
    "rastrigin10D": rastrigin10D
}


def denest_config(cfg: DictConfig, prefix="") -> DictConfig:
    """
    Denest the config object
    """
    denested_cfg = OmegaConf.create()
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            denested_value = denest_config(value, prefix=f"{prefix}{key}.")
            denested_cfg = OmegaConf.merge(denested_cfg, denested_value)
        else:
            denested_cfg[f"{prefix}{key}"] = value
    return denested_cfg



@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    assert cfg.mutation.method == "gaussian", "Only gaussian mutation is supported"
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_file = output_dir / cfg.filename

    crossover = CROSSOVERS[cfg.crossover.method]
    selection = SELECTIONS[cfg.selection.method]
    if cfg.function == "parabolid3D":
        func = FUNCTIONS["parabolid3D"]
        dims = 3
    elif cfg.function == "rastrigin10D":
        func = FUNCTIONS["rastrigin10D"]
        dims = 10
    else:
        raise ValueError("Function not supported")

    records = []
    for r in range(cfg.repeats):
        # Initialize the population
        population = np.random.random((
            cfg.population_size, dims
        ))

        mins = []
        means = []
        maxs = []
        # Run the genetic algorithm
        for t in range(cfg.timesteps):
            parents1, parents2 = Crossover.get_parents(population, cfg.crossover.prob)
            children = crossover(parents1, parents2)
            children = Mutation.gaussian(children, cfg.mutation.sigma, cfg.mutation.prob)

            population = np.concatenate([population, children])
            fitness = func(population)
            population = selection(population, fitness, n=cfg.population_size)

            fitness_new = func(population)
            mins.append(np.min(fitness_new))
            means.append(np.mean(fitness_new))
            maxs.append(np.max(fitness_new))
        

        record = {
            **denest_config(cfg),
            "minimums": mins,
            "means": means,
            "maximums": maxs,
            "best_value": maxs[-1]
        }
        records.append(record)
    
    # Save to a file
    df = pd.DataFrame(records)
    df.to_json(output_file, orient="records", lines=True)







    

if __name__ == "__main__":
    my_app()