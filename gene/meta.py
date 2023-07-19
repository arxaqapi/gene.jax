from functools import partial

import evosax

# import chex
import jax.random as jrd

# import jax.numpy as jnp
from jax import jit, vmap, Array

from gene.core.decoding import DirectDecoder
from gene.core.distances import NNDistanceSimple
from gene.core.models import LinearModel
from gene.learning import learn_gymnax_task, learn_brax_task_untracked


def meta_learn_nn(config: dict):
    """Meta evolution of a neural network parametrized distance function

    - Direct encoding for meta-genotypes
    -

    Instantiate a set of parametrized dfs
    for each df
        evaluate on a curriculum of tasks, see 2023_07_07.md
    use the fitness values to inform the df genome update

    return a learned, parametrized distance function
    """
    rng = jrd.PRNGKey(config["seed"])
    rng, rng_init = jrd.split(rng, 2)

    meta_decoder = DirectDecoder(config)

    meta_strategy = evosax.Sep_CMA_ES(
        popsize=config["evo"]["population_size"], num_dims=meta_decoder.encoding_size()
    )
    meta_state = meta_strategy.initialize(rng_init)

    ask = jit(meta_strategy.ask)
    tell = jit(meta_strategy.tell)

    # Neural network distance network
    nn_dst_model = LinearModel(config["net"]["layer_dimensions"][1:])

    vec_learn_cartpole = jit(
        vmap(
            partial(
                learn_gymnax_task,
                meta_decoder=meta_decoder,
                df_model=nn_dst_model,
                config=config["curriculum"]["cart"],
            ),
            in_axes=(0, None),
        )
    )

    for meta_generation in range(config["evo"]["n_generations"]):
        print(f"[Meta gen n°{meta_generation:>5}]")

        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, meta_state = ask(rng_gen, meta_state)

        # NOTE - Evaluation curriculum
        # NOTE - 1. DF genome to DF using decoder and NNDF, array of df
        # Cannot create an array of DF and pass it down to vectorized functions because
        # an array cannot contain objects (only floats and bools)
        # NOTE - 2. Complete training and evaluation on a curriculum of tasks
        # All distance functions (x) are evaluated by running a complete policy learning loop
        # using GENE with a nn distance function, the sample mean is then evaluated
        f_cp = vec_learn_cartpole(x, rng_eval)
        # NOTE - 3. aggregate fitnesses and weight them
        true_fitness = f_cp  # + 10 * f_hc_100 + 10e1 * f_hc_1000 + 10e2 * f_w2d_1000
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness
        print(f"{true_fitness=}")

        # NOTE - Tell
        meta_state = tell(x, fitness, meta_state)

    return meta_state.mean


# def meta_learn_cgp(config: dict):
#     """Meta evolution of a cgp parametrized distance function"""
#     raise NotImplementedError
