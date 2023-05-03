import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap

from gene.encoding import Encoding_size_function
from gene.evaluate import evaluate_individual

from functools import partial


class PopulationTracker:
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        # self.population_centers = jnp.zeros((config['evo']['population_size'], Encoding_size_function[config['encoding']['type']](config)))
        self.population_centers = [] 
        self.i = 0

    def _compute_mean_pop(self, x):
        return jnp.mean(x, axis=0)

    def center_fitness_per_step(self, rng_eval):
        jit_vmap_evaluate_individual = jit(vmap(partial(evaluate_individual, config=self.config)))

        rng_eval_v = jrd.split(rng_eval, self.config["evo"]["n_generations"])
        res = jit_vmap_evaluate_individual(jnp.array(self.population_centers), rng_eval_v)
        return res


    def update(self, x):
        # self.population_centers = self.population_centers.at[self.i].set(self._compute_mean_pop(x))
        self.population_centers.append(self._compute_mean_pop(x))
        self.i += 1
    
    
