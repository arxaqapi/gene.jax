from jax import jit, vmap, lax, default_backend
import jax.numpy as jnp
import jax.random as jrd
from brax import envs
from brax.envs.wrapper import EpisodeWrapper
import evosax
import wandb

from functools import partial
import json
from time import time

from gene.encoding import Encoding_size_function
from gene.evaluate import evaluate_individual_brax
from gene.tracker import Tracker


def run(config: dict, wdb_run):
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

    env = envs.get_environment(env_name=config["problem"]["environnment"])
    env = EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )
    partial_evaluate_individual = partial(
        evaluate_individual_brax, config=config, env=env
    )
    vmap_evaluate_individual = vmap(partial_evaluate_individual, in_axes=(0, None))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    generation_means = [state.mean]

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
        generation_means.append(state.mean)

    print(f"[Log] - saving all generations @ {time()}")
    # NOTE: Save all generations
    for i, genome in enumerate(generation_means):
        tracker.wandb_save_genome(genome, wdb_run, generation=i)
    print(f"[Log] - end @ {time()}")

    return state


if __name__ == "__main__":
    assert default_backend() == "gpu"

    with open("config/brax_light.json") as f:
        config = json.load(f)

    # seeds = [15684, 253694, 78851363, 148, 9562]

    wdb_run = wandb.init(project="Brax halfcheetah", config=config)
    run(config, wdb_run)
    wdb_run.finish()
