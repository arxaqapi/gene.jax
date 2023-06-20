from typing import Callable

import jax.random as jrd
import evosax

from gene.core.decoding import Decoders
from gene.core.distances import DistanceFunction

# FIXME - merge with experiment
def learning_loop(config: dict,
    df: DistanceFunction, vectorized_eval_f: Callable):
    """Define an evaluation function that can run on batches of genomes and return their fitness.
    Also define the statistics you wanna record.

    This functions is then used in the following learning process.
    """
    rng = jrd.PRNGKey(config["seed"])
    rng, rng_init = jrd.split(rng, 2)

    decoder = Decoders[config["encoding"]["type"]](df)
    decoder.encoding_size()

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=decoder.encoding_size(),
    )

    state = strategy.initialize(rng_init)

    # init tracker
    for generation in config["evo"]["n_generations"]:
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
     
        # TODO - Ask
        x, state = strategy.ask(rng_gen, state)

        # NOTE - Eval
        minimizing_fitness = vectorized_eval_f()
        if config["task"]["maximize"]:
            fitness = -1 * minimizing_fitness
        
        # TODO - Tell
        state = strategy.tell(x, fitness, state)

        # TODO - update tracker
        # - re-evaluate mean individual for fitness
        # - save top genome
        # - save mean genome

    return ()
