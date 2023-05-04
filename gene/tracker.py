import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap

from gene.evaluate import evaluate_individual

from functools import partial


class PopulationTracker:
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.population_centers = []
        self.i = 0

    def _compute_mean_pop(self, x):
        return jnp.mean(x, axis=0)

    def center_fitness_per_step(self, rng_eval):
        jit_vmap_evaluate_individual = jit(
            vmap(partial(evaluate_individual, config=self.config))
        )

        rng_eval_v = jrd.split(rng_eval, self.config["evo"]["n_generations"])
        res = jit_vmap_evaluate_individual(
            jnp.array(self.population_centers), rng_eval_v
        )
        return res

    def update(self, pop_mean):
        self.population_centers.append(pop_mean)
        self.i += 1
