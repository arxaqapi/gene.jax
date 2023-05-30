from jax import jit, vmap, default_backend
import jax.random as jrd
import evosax
import wandb

from functools import partial
import json
from time import time

from gene.encoding import Encoding_size_function
from gene.evaluate import evaluate_individual_brax, get_brax_env
from gene.tracker import Tracker


def run(config: dict, wdb_run):
    assert wdb_run is not None
    rng = jrd.PRNGKey(config["seed"])
    num_dims = Encoding_size_function[config["encoding"]["type"]](config)

    rng, rng_init = jrd.split(rng, 2)
    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=num_dims,
    )

    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng_init, es_params)

    tracker = Tracker(config)
    tracker_state = tracker.init()

    env = get_brax_env(config)
    partial_evaluate_individual = partial(
        evaluate_individual_brax, config=config, env=env
    )
    vmap_evaluate_individual = vmap(partial_evaluate_individual, in_axes=(0, None))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    # ANCHOR - Saving
    tracker.wandb_save_genome(state.mean, wdb_run, now=True)

    for _generation in range(config["evo"]["n_generations"]):
        print(f"[Log] - gen {_generation} @ {time()}")
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        temp_fitness = jit_vmap_evaluate_individual(x, rng_eval)
        fitness = -1 * temp_fitness

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # NOTE - Track metric
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness=temp_fitness,
            mean_ind=state.mean,
            eval_f=partial_evaluate_individual,
            rng_eval=rng_eval,
        )
        tracker.wandb_log(tracker_state, wdb_run)
        # ANCHOR - Saving
        tracker.wandb_save_genome(state.mean, wdb_run, now=True)

    print(f"[Log] - end @ {time()}")

    return state


if __name__ == "__main__":
    assert default_backend() == "gpu"

    with open("config/brax_sep_cmaes.json") as f:
        config = json.load(f)

    seeds = [15684, 253694, 78851363, 148, 9562]
    config["seed"] = seeds[0]

    wdb_run = wandb.init(project="Brax halfcheetah", config=config)

    run(config, wdb_run)

    wdb_run.finish()
