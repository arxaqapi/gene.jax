from functools import partial

from jax import jit, vmap
import jax.random as jrd
import evosax

from gene.core.models import BoundedLinearModelConf
from gene.core.decoding import Decoders, Decoder
from gene.core.distances import DistanceFunction
from gene.core.evaluation import get_brax_env, rollout_brax_task
from gene.v1.tracker import Tracker


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


def learn_brax_task(config: dict, df: DistanceFunction, wdb_run):
    """Run a es training loop specifically tailored for brax tasks.

    Args:
        config (dict): config of the current run.
        df (DistanceFunction): Distance function to use, can be parametrized.
        wdb_run: wandb run object used to log

    Returns:
        _type_: _description_
    """
    assert wdb_run is not None
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
            fitnesses=true_fitness,
            mean_ind=state.mean,
            eval_f=eval_f,
            rng_eval=rng_eval,
        )
        if wdb_run is not None:
            tracker.wandb_log(tracker_state, wdb_run)
            # if (_generation + 1) % 100 == 0:
            tracker.wandb_save_genome(
                genome=state.mean,
                wdb_run=wdb_run,
                file_name=f"g{str(_generation).zfill(3)}_mean_indiv",
                now=True,
            )
    # NOTE - Save best at end of run
    if wdb_run is not None:
        tracker.wandb_save_genome(
            genome=state.best_member,
            wdb_run=wdb_run,
            file_name="final_best_indiv",
            now=True,
        )

    return tracker_state
