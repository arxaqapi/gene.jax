from pathlib import Path
import time
from functools import partial

import jax.random as jrd
from jax import jit, vmap, default_backend
import evosax

from gene.evaluate import evaluate_individual
from gene.encoding import Encoding_size_function
from gene.tracker import Tracker


def run(
    config: dict,
    wdb_run,
):
    rng = jrd.PRNGKey(config["seed"])
    tracker = Tracker(config)
    tracker_state = tracker.init()

    num_dims = Encoding_size_function[config["encoding"]["type"]](config)

    rng, rng_init = jrd.split(rng, 2)
    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=num_dims,
    )

    # NOTE: Sampled from uniform distribution
    state = strategy.initialize(rng_init)

    vmap_evaluate_individual = vmap(
        partial(evaluate_individual, config=config), in_axes=(0, None)
    )
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    for generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state)
        # NOTE - Evaluate
        temp_fitness = jit_vmap_evaluate_individual(x, rng_eval)
        fitness = -1.0 * temp_fitness
        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state)

        # NOTE - Track metrics
        tracker_state = tracker.update(tracker_state, None, temp_fitness)
        tracker.wandb_log(tracker_state, wdb_run)
    return state


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

    args = parser.parse_args()

    config_file = Path(args.config)
    if config_file.exists():
        with config_file.open() as f:
            config = json.load(f)
    else:
        raise ValueError("No config file found")

    import wandb

    wdb_run = wandb.init(project="Cartpole", config=config)

    run(config, wdb_run)
