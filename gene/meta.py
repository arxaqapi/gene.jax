from functools import partial

import evosax
import jax.random as jrd
from jax import jit, vmap, Array

from gene.core.decoding import DirectDecoder, Decoder
from gene.core.distances import NNDistance
from gene.learning import learn_gymnax_task, learn_brax_task_untracked


def get_nn_df(x: Array, meta_decoder: Decoder, config: dict) -> NNDistance:
    """takes an direct-encoded neural network, decodes it
    and makes it a distance function"""

    df_phenotype = meta_decoder.decode(x)
    return NNDistance(
        df_phenotype, config, nn_layers_dims=config["net"]["layer_dimensions"]
    )


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

    vec_get_nn_df = vmap(partial(get_nn_df, meta_decoder=meta_decoder, config=config))
    vec_learn_cartpole = vmap(
        partial(learn_gymnax_task, config=config["curriculum"]["cart"]), in_axes=(0, None)
    )
    vec_learn_hc_100 = vmap(
        partial(learn_brax_task_untracked, config=config["curriculum"]["hc_100"]),
        in_axes=(0, None),
    )
    vec_learn_hc_1000 = vmap(
        partial(learn_brax_task_untracked, config=config["curriculum"]["hc_1000"]),
        in_axes=(0, None),
    )

    for meta_generation in range(config["evo"]["n_generations"]):
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, meta_state = ask(rng_gen, meta_state)

        # NOTE - Evaluation curriculum
        # NOTE - 1. DF genome to DF using decoder and NNDF, array of df
        dfs: Array = vec_get_nn_df(x)

        # NOTE - 2. Complete training and evaluation on a curriculum of tasks
        f_cp = vec_learn_cartpole(dfs, rng_eval)
        f_hc_100 = vec_learn_hc_100(dfs, rng_eval) if f_cp > 400 else 0
        f_hc_1000 = vec_learn_hc_1000(dfs, rng_eval) if f_hc_100 > 200 else 0

        # NOTE - 3. aggregate fitnesses and weight them
        true_fitness = f_cp + 10 * f_hc_100 + 10e1 * f_hc_1000  # + 10e2 * f_w2d_1000
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        meta_state = tell(x, fitness, meta_state)

    return meta_state.mean


# def meta_learn_cgp(config: dict):
#     """Meta evolution of a cgp parametrized distance function"""
#     raise NotImplementedError
