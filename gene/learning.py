from functools import partial

from jax import jit, vmap, Array
import jax.numpy as jnp
import jax.random as jrd
import evosax

from gene.tracker import Tracker, TrackerState
from gene.core.models import Models
from gene.core.decoding import Decoders, Decoder
from gene.core.distances import DistanceFunction
from gene.core.evaluation import get_braxv1_env, rollout_brax_task


def brax_eval(genome: Array, rng: jrd.KeyArray, decoder: Decoder, config: dict, env):
    model_parameters = decoder.decode(genome)
    model = Models[config["net"]["architecture"]](config)

    fitness = rollout_brax_task(
        config=config,
        model=model,
        model_parameters=model_parameters,
        env=env,
        rng_reset=rng,
    )

    return fitness


def brax_eval_n_times(
    genome: Array,
    rng: jrd.KeyArray,
    decoder: Decoder,
    config: dict,
    env,
    number_evaluations: int = 10,
):
    """This evaluates a single genome 0 times with different seeds and returns
    the median over the collected returns.

    Doing so forces the learning process to generalize to multiple envs, and not be
    stuck in a fragile exploitation setting.

    Args:
        genome (Array): The genome of the evaluated individual.
        rng (jrd.KeyArray): rng key used to evaluate
        decoder (Decoder): The decoder object used to decode the individuals `genome`.
        config (dict): The config dict of the current run.
        env (_type_): The brax environment used to evaluate the individual.
        number_evaluations (int, optional): Number of time the individual
            is evaluated. Defaults to 10.
    """
    rngs = jrd.split(rng, number_evaluations)
    eval_f = partial(brax_eval, decoder=decoder, config=config, env=env)
    fitnesses = jit(vmap(eval_f, in_axes=(None, 0)))(genome, rngs)

    # take median fitness and returns it
    return jnp.median(fitnesses)


def learn_brax_task(
    config: dict, df: DistanceFunction, wdb_run
) -> tuple[Tracker, TrackerState]:
    """Run an es training loop specifically tailored for brax tasks.

    Args:
        config (dict): config of the current run.
        df (DistanceFunction): Distance function to use, can be parametrized.
        wdb_run: wandb run object used to log

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

    env = get_braxv1_env(config)

    # Each individual is evaluated a single time or multiple times in parallel
    evaluation_f = (
        brax_eval_n_times if config["evo"]["n_evaluations"] > 1 else brax_eval
    )

    partial_eval_f = partial(evaluation_f, decoder=decoder, config=config, env=env)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    tracker = Tracker(config)
    tracker_state = tracker.init()

    # NOTE - save first individual
    if wdb_run is not None:
        tracker.wandb_save_genome(state.mean, wdb_run, "initial_best_indiv", now=True)
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

        # NOTE - Track metrics
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=x,
            fitnesses=true_fitness,
            mean_ind=state.mean,
            eval_f=partial_eval_f,
            rng_eval=rng_eval,
        )
        if wdb_run is not None:
            tracker.wandb_log(tracker_state, wdb_run)
            # Saves only every 100 generation
            if (_generation + 1) % 100 == 0:
                tracker.wandb_save_genome(
                    genome=state.mean,
                    wdb_run=wdb_run,
                    file_name=f"g{str(_generation).zfill(3)}_mean_indiv",
                    now=True,
                )
    # NOTE - Save best individuals at end of run
    if wdb_run is not None:
        # Overall best individuals
        for i, top_k_indiv in enumerate(tracker_state["backup"]["top_k_individuals"]):
            tracker.wandb_save_genome(
                genome=top_k_indiv,
                wdb_run=wdb_run,
                file_name=f"final_top_{i}_indiv",
                now=True,
            )

    return tracker, tracker_state
