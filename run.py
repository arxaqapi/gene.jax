from pathlib import Path
import logging
import time
from functools import partial

import jax.random as jrd
from jax import jit, vmap, default_backend
import evosax

from gene.evaluate import evaluate_individual
from gene.encoding import Encoding_size_function


def run(
    settings: dict,
    rng: jrd.KeyArray = jrd.PRNGKey(5),
):
    logger = logging.getLogger("logger")

    num_dims = Encoding_size_function[settings["encoding"]["type"]](settings)

    rng, rng_init = jrd.split(rng, 2)
    strategy = evosax.Strategies[settings["evo"]["strategy_name"]](
        popsize=settings["evo"]["population_size"],
        num_dims=num_dims,
    )

    fit_shaper = evosax.FitnessShaper(maximize=settings["problem"]["maximize"])
    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng_init, es_params)

    vmap_evaluate_individual = vmap(partial(evaluate_individual, settings=settings))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    for _generation in range(settings["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # Here, each individual has an unique random key used for evaluation purposes
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

    # TODO: load a config file passed as argument
    import json

    config_file = Path("config/base_cartpole.json")
    if config_file.exists():
        with config_file.open() as f:
            settings = json.load(f)

    log_path = Path("log")
    if not log_path.exists():
        log_path.mkdir()

    logger = logging.getLogger("logger")
    f_handler = logging.FileHandler(
        filename=log_path.absolute()
        / f"{int(time.time())}_run{settings['encoding']['type']}.log",
        mode="a",
    )
    f_handler.setFormatter(
        logging.Formatter(
            fmt="%(levelname)-8s %(asctime)s \t %(filename)s @f %(funcName)s @L%(lineno)s - %(message)s",
            datefmt="%Y/%m/%d_%H.%M.%S",
        )
    )
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)

    run(settings)
