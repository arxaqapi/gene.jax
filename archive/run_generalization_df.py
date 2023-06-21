"""
Use the learned distance function in: arxaqapi/Distance-tests/jcks0qdx 
to evaluate the networks.

Train with theses learned distance functions
"""
from functools import partial
from copy import deepcopy
from pathlib import Path
from time import time
import json

import jax.numpy as jnp
import jax.random as jrd
from jax import vmap, jit, default_backend
import flax.linen as nn
import evosax
import wandb

from gene.learn_distance import NNDistance, get_brax_env
from gene.encoding import (
    _direct_enc_genome_size,
    gene_decoding_w_dist,
    gene_enc_genome_size,
)
from gene.network import BoundedLinearModel
from gene.evaluate import _rollout_brax
from gene.tracker import Tracker
from run_brax import run


def get_learned_distance_f(config: dict):
    """arxaqapi/Distance-tests/jcks0qdx"""
    import wandb

    api = wandb.Api()
    # config = api.run("arxaqapi/Distance-tests/jcks0qdx").config
    # assert config and config["distance_network"]["layer_dimensions"] are the same

    artifact = api.artifact(
        "arxaqapi/Distance-tests/best_member_model_parameters:latest"
    )

    distance = NNDistance(
        distance_genome=jnp.zeros(
            (_direct_enc_genome_size(config["distance_network"]["layer_dimensions"]))
        ),
        layer_dimensions=config["distance_network"]["layer_dimensions"],
    )
    distance.load_parameters(Path(artifact.download()) / "best_member")

    return distance


# FIXME - code repetition
def evaluate_individual_brax_w_distance_fitness(
    genome: jnp.array, rng: jrd.KeyArray, config: dict, env, distance: NNDistance
) -> tuple[float, nn.FrozenDict]:
    """Evaluates a single individual `genome` using a parametrized
    distance function `distance`.


    Args:
        genome (jnp.array): _description_
        rng (jrd.KeyArray): _description_
        config (dict): _description_
        env (_type_): _description_
        distance (NNDistance): _description_

    Returns:
        tuple[float, nn.FrozenDict]: the fitness as a float
            and the `model_parameters` of the decoded genome.
    """
    model = BoundedLinearModel(config["net"]["layer_dimensions"][1:])
    model_parameters: nn.FrozenDict = nn.FrozenDict(
        {
            "params": gene_decoding_w_dist(
                genome, config=config, distance_network=distance
            )
        }
    )

    # print(tree_util.tree_map(lambda x: x.shape, model_parameters))
    fitness = _rollout_brax(
        model=model,
        model_parameters=model_parameters,
        config=config,
        env=env,
        rng_reset=rng,
    )
    return fitness


def learn_task_with_df(config: dict, distance: NNDistance, wdb_run):
    # NOTE - Init
    assert wdb_run is not None
    rng = jrd.PRNGKey(config["seed"])
    # We use GENE as an encoding with a learned distance function
    assert config["encoding"]["type"] == "gene"
    num_dims = gene_enc_genome_size(config)

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
    _partial_evaluate_individual = partial(
        evaluate_individual_brax_w_distance_fitness,
        config=config,
        env=env,
        distance=distance,
    )
    jit_vmap_evaluate_individual = jit(
        vmap(_partial_evaluate_individual, in_axes=(0, None))
    )

    # NOTE - Ask/Eval/Tell Loop
    for _generation in range(config["evo"]["n_generations"]):
        print(f"[Log] - gen {_generation} @ {time()}")

        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        temp_fitness = jit_vmap_evaluate_individual(x, rng_eval)
        fitness = -1 * temp_fitness  # we want to maximize the objective f.

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # NOTE - Track metric
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitnesses=temp_fitness,
            mean_ind=state.mean,
            eval_f=_partial_evaluate_individual,
            rng_eval=rng_eval,
        )
        tracker.wandb_log(tracker_state, wdb_run)
        # ANCHOR - Saving
        tracker.wandb_save_genome(state.mean, wdb_run, now=True)

    print(f"[Log] - end @ {time()}")

    return state


if __name__ == "__main__":
    assert default_backend() == "gpu"

    # check config for each new run
    with open("config/brax_benchmark.json") as f:
        base_config = json.load(f)

    # https://github.com/google/brax/blob/main/brax/envs/__init__.py#L46
    benchmark_configs = {
        "halfcheetah": {
            "in": 18,
            "out": 6,
        },
        "humanoid": {
            "in": 240,
            "out": 8,
        },
        "ant": {
            "in": 87,
            "out": 8,
        },
        "walker2d": {
            "in": 17,
            "out": 6,
        },
    }
    # Init new run
    # Run
    # close run

    for task, layer_info in benchmark_configs.items():
        config = deepcopy(base_config)
        config["task"]["environnment"] = task
        config["net"]["layer_dimensions"] = (
            [layer_info["in"]] + [128, 128] + [layer_info["out"]]
        )

        wdb_run = wandb.init(
            project="Brax bench",
            config=config,
            tags=[config["task"]["environnment"], "from-scratch"],
        )
        # learn_task_with_df(
        #     config=config,
        #     distance=get_learned_distance_f(config),
        #     wdb_run=wdb_run,
        # )

        run(config, wdb_run)

        wdb_run.finish()
