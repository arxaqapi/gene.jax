from gene.evaluatax import evaluate_models, evaluate_model, evaluate_flatnet
from gene.deepax import FlatNet
from gene.distances import tag_gene

import jax
import jax.numpy as jnp
from evosax import SNES, xNES, CMA_ES
from evosax import FitnessShaper
import evosax
import gymnasium as gym
from tqdm import tqdm

from functools import partial
import logging
from uuid import uuid1
from pathlib import Path


def run_expe(number_neurons: int = 1096, algorithm: str = 'SNES', pop_size: int = 1, max_gen: int = 1, logger: logging.Logger =None):
    logger = logging.getLogger("logger")
    env = gym.make('ALE/SpaceInvaders-v5', obs_type='ram', full_action_space=True)

    fit_shaper = FitnessShaper(maximize=True)
    rng = jax.random.PRNGKey(0)

    strategy = evosax.Strategies[algorithm](
        popsize=pop_size,
        num_dims=number_neurons # 1096
    )
    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng, es_params)

    # vmap_evaluation_f = jax.vmap(partial(evaluate_flatnet, env))

    for generation in tqdm(range(max_gen)):
        logger.info(f'[INFO - {algorithm}]: Generation {generation + 1} is starting')

        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        # Ask generation | x (popsize, num_dims)
        x, state = strategy.ask(rng_gen, state, es_params)
        # Evaluate generation[s] genome
        temp_fitness = evaluate_models(models=[FlatNet(genome, distance_f=tag_gene) for genome in x], env=env)
        # temp_fitness = vmap_evaluation_f(x) # vmap over the rows (each row in parrallel)
        
        # NOTE: after fitness evaluation, apply fitness reshaper: https://github.com/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb
        fitness = fit_shaper.apply(x, temp_fitness)
        
        # Tell / Update step
        state = strategy.tell(x, fitness, state, es_params)

        if (generation+1) % 10 == 0:
            logger.info(f'[INFO - {algorithm}]: Gen: {generation+1} | Fitness {state.best_fitness:.5f}')
            # state.best_member

    env.close() 
    return state.best_member


if __name__ == "__main__":
    logpath = Path('log')
    if not logpath.exists():
        logpath.mkdir()
    # NOTE: add HPO to logging info (extend class and stuff)
    logger = logging.getLogger("logger")
    fhandler = logging.FileHandler(filename=f'/home/gene.jax/log/run_log_{str(uuid1())}.log', mode='a')
    formatter = logging.Formatter('[%(levelname)s - %(asctime)s]: %(name)s, %(message)s', '%Y/%m/%d_%H.%M.%S')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    

    run_expe()