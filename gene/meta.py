import evosax
import jax.random as jrd
from jax import jit, vmap, Array

from gene.core.decoding import DirectDecoder, Decoder
from gene.core.distances import NNDistance
from gene.core.evaluation import get_braxv1_env


def meta_nn_evaluation(df_genome: Array, meta_decoder: Decoder, rng_eval: jrd.KeyArray, config: dict) -> float:
    # 1. DF genome to DF using decoder and NNDF
    df_phenotype = meta_decoder.decode(df_genome)
    df = NNDistance(df_phenotype, config)
    # 2. Evaluate on a curriculum of tasks
    
    # 3. 
    fitness = 0.0
    return fitness


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

    # FIXME - get from config
    POP_SIZE = 128
    META_GENERATIONS = 1000
    DIMENSIONS = meta_decoder.encoding_size()

    meta_strategy = evosax.Sep_CMA_ES(popsize=POP_SIZE, num_dims=DIMENSIONS)
    meta_state = meta_strategy.initialize(rng_init)

    ask = jit(meta_strategy.ask)
    tell = jit(meta_strategy.tell)
    # partial_eval_f = partial(evaluation_f, decoder=decoder, config=config, env=env)
    # vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    for meta_generation in range(META_GENERATIONS):
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, meta_state = ask(rng_gen, meta_state)

        # NOTE - Evaluate
        true_fitness = meta_nn_evaluation(x, rng_eval)
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        meta_state = tell(x, fitness, meta_state)

    return meta_state.mean


# def meta_learn_cgp(config: dict):
#     """Meta evolution of a cgp parametrized distance function"""
#     raise NotImplementedError
