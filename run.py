from pathlib import Path
import logging
import time
from functools import partial

import jax.random as jrd
from jax import jit, vmap, default_backend
import evosax

from gene.evaluate import evaluate_individual
from gene.encoding import Encoding_size_function
from gene.tracker import PopulationTracker


def run(
    config: dict,
    rng: jrd.KeyArray = jrd.PRNGKey(5),
):
    logger = logging.getLogger("logger")
    tracker = PopulationTracker(config)

    num_dims = Encoding_size_function[config["encoding"]["type"]](config)

    rng, rng_init = jrd.split(rng, 2)
    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=num_dims,
    )

    # NOTE: Check if uniform or normal distr
    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng_init, es_params)

    # Enable logging data during training process
    es_logging = evosax.ESLog(
        num_dims=num_dims,
        num_generations=config["evo"]["n_generations"],
        top_k=5,
        maximize=True,
    )
    log = es_logging.initialize()

    vmap_evaluate_individual = vmap(
        partial(evaluate_individual, config=config), in_axes=(0, None)
    )
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    for generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        temp_fitness = jit_vmap_evaluate_individual(x, rng_eval)
        fitness = -1.0 * temp_fitness

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # Log / stats step: Add the fitness to log object
        tracker.update(state.mean)
        log = es_logging.update(log, x, temp_fitness)
        logger.info(
            "Generation: ", generation, "Performance: ", log["log_top_1"][generation]
        )
    return state, log, tracker.center_fitness_per_step(rng)


if __name__ == "__main__":
    assert default_backend() == "gpu"

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Start the experimen described in the config file"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        default="config/base_cartpole.json",
        help="Config file to use",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Activate or deactivate the logging capabilities",
    )
    # , required=False, default=True
    args = parser.parse_args()

    config_file = Path(args.config)
    if config_file.exists():
        with config_file.open() as f:
            config = json.load(f)
    else:
        raise ValueError("No config file found")

    log_path = Path("log")
    if not log_path.exists():
        log_path.mkdir()

    logger = logging.getLogger("logger")
    f_handler = logging.FileHandler(
        filename=log_path.absolute()
        / f"{int(time.time())}_run_{config['encoding']['type']}.log",
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
    if args.verbose == False:
        logger.disabled = not args.verbose

    run(config)
