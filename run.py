from pathlib import Path
import logging
import time
from functools import partial

import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap, default_backend
import evosax
from tqdm import tqdm

from gene.evaluate import evaluate_individual


def run(
    settings: dict,
    rng: jrd.KeyArray = jrd.PRNGKey(0),
):
    logger = logging.getLogger("logger")

    rng, rng_init = jrd.split(rng, 2)
    # FIXME: no bias for the moment: add with n(d + 1): sum() * (settings['d'] + 1)
    strategy = evosax.Strategies[settings["evo"]["strategy_name"]](
        popsize=settings["evo"]["population_size"],
        num_dims=sum(settings["net"]["layer_dimensions"]),
    )
    fit_shaper = evosax.FitnessShaper(maximize=settings["problem"]["maximize"])
    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng_init, es_params)

    # FIXME: uses the same key for each generation
    jit_vmap_evaluate_individual = jit(
        vmap(partial(evaluate_individual, settings=settings))
    )

    for _generation in tqdm(
        range(settings["evo"]["n_generations"]), desc="Generation loop", position=0
    ):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        rng_eval_v = jrd.split(rng_eval, settings["evo"]["population_size"])
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        temp_fitness = jit_vmap_evaluate_individual(x, rng_eval_v)
        fitness = fit_shaper.apply(x, temp_fitness)

        logger.info(f"[[{_generation}]] - {temp_fitness=}")
        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # Log / stats step
        # ...
    return state.best_fitness, state.best_member


if __name__ == "__main__":
    assert default_backend() == "gpu"

    log_path = Path("log")
    if not log_path.exists():
        log_path.mkdir()

    logger = logging.getLogger("logger")
    f_handler = logging.FileHandler(
        filename=log_path.absolute() / f"{int(time.time())}_run.log", mode="a"
    )
    f_handler.setFormatter(
        logging.Formatter(
            fmt="%(levelname)-8s %(asctime)s \t %(filename)s @f %(funcName)s @L%(lineno)s - %(message)s",
            datefmt="%Y/%m/%d_%H.%M.%S",
        )
    )
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)

    settings = {
        "d": 1,
        "evo": {
            "strategy_name": "OpenES",
            "n_generations": 10,
            "population_size": 20,
        },
        "net": {
            "layer_dimensions": [4, 32, 2],  # Infer network config from dimensions
        },
        "problem": {
            "environnment": "CartPole-v1",
            "maximize": True,
        },
    }

    run(settings)
