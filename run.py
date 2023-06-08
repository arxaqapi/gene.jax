from functools import partial
import argparse
import json

import jax.random as jrd
from jax import jit, vmap, default_backend
import evosax
import wandb

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

    partial_evaluate_individual = partial(evaluate_individual, config=config)
    vmap_evaluate_individual = vmap(partial_evaluate_individual, in_axes=(0, None))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    tracker.wandb_save_genome(state.mean, wdb_run, True)

    for _generation in range(config["evo"]["n_generations"]):
        print(f"[Gen {_generation}]")
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
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitnesses=temp_fitness,
            mean_ind=state.mean,
            eval_f=partial_evaluate_individual,
            rng_eval=rng_eval,
        )
        tracker.wandb_log(tracker_state, wdb_run)
        tracker.wandb_save_genome(state.mean, wdb_run, True)

    return state


if __name__ == "__main__":
    assert default_backend() == "gpu"

    parser = argparse.ArgumentParser(
        description="Start the experiment described in the config file"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        default="config/base_cartpole.json",
        help="Config file to use",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    for seed in range(10):
        config["seed"] = seed
        config["encoding"]["type"] = "direct"

        wdb_run = wandb.init(project="Cartpole", config=config)
        run(config, wdb_run)
        wdb_run.finish()
