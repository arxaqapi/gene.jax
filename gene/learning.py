from functools import partial

from jax import jit, vmap
import jax.random as jrd
import evosax

from gene.core.models import BoundedLinearModelConf
from gene.core.decoding import Decoders, Decoder
from gene.core.distances import DistanceFunction
from gene.core.evaluation import get_brax_env, rollout_brax_task


def brax_eval(genome, rng, decoder: Decoder, config: dict, env):
    model_parameters = decoder.decode(genome)
    model = BoundedLinearModelConf(config)

    fitness = rollout_brax_task(
        config=config,
        model=model,
        model_parameters=model_parameters,
        env=env,
        rng_reset=rng,
    )

    return fitness


# FIXME - merge with experiment
def learn_brax_task(
    config: dict,
    df: DistanceFunction,
):
    """Run a es training loop specifically tailored for brax tasks.

    Args:
        config (dict): config of the current run.
        df (DistanceFunction): Distance function to use, can be parametrized.

    Returns:
        _type_: _description_
    """
    rng = jrd.PRNGKey(config["seed"])
    rng, rng_init = jrd.split(rng, 2)

    decoder = Decoders[config["encoding"]["type"]](config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=decoder.encoding_size(),
    )

    state = strategy.initialize(rng_init)

    env = get_brax_env(config)

    eval_f = partial(brax_eval, decoder=decoder, config=config, env=env)
    vectorized_eval_f = jit(vmap(eval_f, in_axes=(0, None)))

    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    # init tracker
    for _generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)

        # NOTE - Ask
        x, state = ask(rng_gen, state)

        # NOTE - Eval
        true_fitness = vectorized_eval_f(x, rng_eval)
        if config["task"]["maximize"]:
            fitness = -1 * true_fitness

        # NOTE - Tell
        state = tell(x, fitness, state)

        # TODO - update tracker
        # - re-evaluate mean individual for fitness
        print(eval_f(state.mean, rng_eval))
        # - save top genome
        # - save mean genome

    return ()
