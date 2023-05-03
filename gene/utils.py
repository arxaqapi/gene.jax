from functools import partial

import jax.random as jrd
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from gene.evaluate import evaluate_individual


def plot(*args, info: str = ""):
    """Simple plotting utility function.
    *args: is sequence of tuples of size 3. Each tuple contains the mean value vector, its standart deviation and a label
    """
    plt.style.use("bmh")  # fivethirtyeight
    plt.figure(figsize=(12, 6))
    for m, s, label in args:
        plt.fill_between(range(len(m)), m + 0.5 * s, m - 0.5 * s, alpha=0.35)
        plt.plot(m, label=label)
    plt.xlabel("n° generations")
    plt.ylabel("fitness")
    plt.title(f"Mean fitness over generations\n{info}")
    plt.legend()
    plt.show()



def evaluate_mean_population(x, config: dict, rng: jrd.KeyArray = jrd.PRNGKey(0)):
    """x: (batch, d)"""
    # Retrieve the theoretical individual at the center of the population
    center = jnp.mean(x, axis=0)
    # Evaluate the individual
    fitness = evaluate_individual(center, rng, config=config)
    return fitness


def evaluate_bench(run_f, config: dict, rng: jrd.KeyArray = jrd.PRNGKey(0), n: int = 10):
    rng_run = jrd.split(rng, n)
    vmap_run = vmap(partial(run_f, config=config))
    _, logs, center_fitness = vmap_run(rng=rng_run)
    return {
        "fit_best": logs["log_gen_1"].mean(axis=0),
        "fit_best_std": logs["log_gen_1"].std(axis=0),
        "mean_fit": logs["log_gen_mean"].mean(axis=0),
        "mean_fit_std": logs["log_gen_std"].mean(axis=0),
        "logs": logs,
        "population_center": center_fitness.mean(axis=0),
        "population_std": center_fitness.std(axis=0),
    }