from functools import partial
import argparse
import json

import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap, default_backend
import evosax
import wandb

from gene.evaluate import evaluate_individual_soft
from gene.encoding import Encoding_size_function
from gene.tracker import Tracker


def run(
    config: dict,
    wdb_run,
):
    rng = jrd.PRNGKey(config["seed"])
    rng_action_sampling = jrd.PRNGKey(0)

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

    partial_evaluate_individual = partial(
        evaluate_individual_soft, config=config, rng_action_sampling=rng_action_sampling
    )
    vmap_evaluate_individual = vmap(partial_evaluate_individual, in_axes=(0, None))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    generation_means = [state.mean]

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
        generation_means.append(state.mean)

    # NOTE: Save all generations
    for i, genome in enumerate(generation_means):
        tracker.wandb_save_genome(genome, wdb_run, generation=i)
    return state


def run_multi_eval(config: dict, wdb_run, k: int = 10):
    rng = jrd.PRNGKey(config["seed"])
    rng_action_sampling = jrd.PRNGKey(0)

    tracker = Tracker(config)
    tracker_state = tracker.init()

    num_dims = Encoding_size_function[config["encoding"]["type"]](config)

    rng, rng_init = jrd.split(rng, 2)
    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=num_dims,
    )

    # NOTE: Sampled from uniform distribution
    es_params = strategy.default_params
    es_params = es_params.replace(sigma_init=0.1)
    state = strategy.initialize(rng_init, es_params)

    partial_evaluate_individual = partial(
        evaluate_individual_soft, config=config, rng_action_sampling=rng_action_sampling
    )
    vmap_evaluate_individual = vmap(partial_evaluate_individual, in_axes=(0, None))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    generation_means = [state.mean]

    for _generation in range(config["evo"]["n_generations"]):
        print(f"[Gen {_generation}]")
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        rng_ks = jrd.split(rng_eval, k)
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state)
        # NOTE - Evaluate k times

        k_fitnesses = jnp.zeros((k, config["evo"]["population_size"]))
        for i, rng_k in enumerate(rng_ks):
            # eval individuals
            temp_fitness = jit_vmap_evaluate_individual(x, rng_k)
            corrected_fitness = -1.0 * temp_fitness
            k_fitnesses = k_fitnesses.at[i].set(corrected_fitness)

        # mean over fitness & use this as new fitness
        fitness = k_fitnesses.mean(axis=0)

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
        generation_means.append(state.mean)

    # NOTE: Save all generations
    for i, genome in enumerate(generation_means):
        tracker.wandb_save_genome(genome, wdb_run, generation=i)
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

    # for seed in range(10):
    #     config["seed"] = seed
    config["encoding"]["type"] = "direct"

    wdb_run = wandb.init(project="Cartpole Soft", config=config)
    run_multi_eval(config, wdb_run)
    wdb_run.finish()
