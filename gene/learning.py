from functools import partial
from datetime import datetime

from jax import jit, vmap, Array
import jax.numpy as jnp
import jax.random as jrd
import evosax
import flax.linen as nn

from gene.tracker import Tracker, TrackerState
from gene.core.models import Models
from gene.core.decoding import Decoder, get_decoder
from gene.core.distances import DistanceFunction, NNDistanceSimple, CGPDistance
from gene.core.evaluation import get_braxv1_env, rollout_brax_task, rollout_gymnax_task
from gene.timer import Timer


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
    config: dict, df: DistanceFunction, wdb_run, save_step: int = 2000
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

    decoder = get_decoder(config)(config, df)

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

    tracker = Tracker(config, decoder)
    # if save_step > config["evo"]["n_generations"] then skip savng mean indiv
    tracker_state = tracker.init(
        skip_mean_backup=save_step >= config["evo"]["n_generations"]
    )

    # NOTE - save first individual
    if wdb_run is not None:
        tracker.wandb_save_genome(state.mean, wdb_run, "initial_mean_indiv", now=True)
        tracker_state = tracker.set_initial_mean(tracker_state, state.mean)

    total_timer = Timer()
    evaluation_timer = Timer()
    total_timer.start()
    for _generation in range(config["evo"]["n_generations"]):
        print(
            f"[Log] - Generation n° {_generation:>6}"
            + f"@ {datetime.now().strftime('%Y.%m.%d_%H:%M:%S')}"
        )
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)

        # NOTE - Ask
        x, state = ask(rng_gen, state)

        # NOTE - Eval
        evaluation_timer.start()

        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        evaluation_timer.stop()
        print(
            f"[Log] - {str(evaluation_timer)} "
            + f"for {config['evo']['population_size']} parallel evaluations"
        )
        evaluation_timer.reset()

        # NOTE - Tell
        state = tell(x, fitness, state)

        # NOTE - Track metrics
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=x,
            fitnesses=true_fitness,
            sample_mean=state.mean,
            eval_f=partial_eval_f,
            rng_eval=rng_eval,
            skip_mean_backup=save_step >= config["evo"]["n_generations"],
        )
        if wdb_run is not None:
            tracker.wandb_log(tracker_state, wdb_run)
            # Saves only every 'save_step' generation
            if (_generation + 1) % save_step == 0:
                tracker.wandb_save_genome(
                    genome=state.mean,
                    wdb_run=wdb_run,
                    file_name=f"g{str(_generation).zfill(3)}_mean_indiv",
                    now=True,
                )
    total_timer.stop()
    print(f"[Log] - Generation loop: {str(total_timer)}")
    # NOTE - Save mean and best individuals at end of run
    if wdb_run is not None:
        # Mean
        tracker.wandb_save_genome(state.mean, wdb_run, "final_mean_indiv", now=True)
        tracker_state = tracker.set_final_mean(tracker_state, state.mean)
        # Overall best individuals
        for i, top_k_indiv in enumerate(tracker_state["backup"]["top_k_individuals"]):
            tracker.wandb_save_genome(
                genome=top_k_indiv,
                wdb_run=wdb_run,
                file_name=f"final_top_{i}_indiv",
                now=True,
            )

    return tracker, tracker_state


def learn_brax_task_cgp(
    cgp_genome: Array,
    rng: jrd.KeyArray,
    config: dict,
    cgp_config: dict,
) -> float:
    """Run an es training loop specifically tailored for brax tasks
    using a cgp defined function.

        Returns:
        float: fitness of the overall best individual encountered
    """
    rng, rng_init = jrd.split(rng, 2)

    df = CGPDistance(cgp_genome=cgp_genome, cgp_config=cgp_config)
    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=decoder.encoding_size(),
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    # Each individual is evaluated a single time or multiple times in parallel
    evaluation_f = (
        brax_eval_n_times if config["evo"]["n_evaluations"] > 1 else brax_eval
    )

    env = get_braxv1_env(config)
    partial_eval_f = partial(evaluation_f, decoder=decoder, config=config, env=env)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    overall_best_member = {"individual": state.mean, "fitness": 0}

    for _generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)

        # NOTE - Ask
        x, state = ask(rng_gen, state)

        # NOTE - Eval
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        state = tell(x, fitness, state)

        # NOTE - update best
        all_members = jnp.vstack((x, overall_best_member["individual"]))
        all_fitnesses = jnp.hstack((true_fitness, overall_best_member["fitness"]))
        best_member_i = jnp.argmax(all_fitnesses)
        # Overwrite best member
        overall_best_member = {
            "individual": all_members[best_member_i],
            "fitness": all_fitnesses[best_member_i],
        }

    return overall_best_member["fitness"]


# ================================================
# ============   NN Learning   ==================
# ================================================


def learn_brax_task_untracked_nn_df(
    df_genotype: Array,
    rng: jrd.KeyArray,
    meta_decoder: Decoder,
    df_model: nn.Module,
    config: dict,
) -> float:
    """Run an es training loop specifically tailored for brax tasks.

    Args:
        config (dict): config of the current run.
        df (DistanceFunction): Distance function to use, can be parametrized.
        wdb_run: wandb run object used to log

    Returns:
        float: fitness of the overall best individual encountered
    """
    rng, rng_init = jrd.split(rng, 2)

    # NOTE - The distance function genotype needs to be decoded here
    # because objects cannot be given as input of vectorized functions
    df_phenotype = meta_decoder.decode(df_genotype)
    df = NNDistanceSimple(model_parameters=df_phenotype, model=df_model)
    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=decoder.encoding_size(),
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    # Each individual is evaluated a single time or multiple times in parallel
    evaluation_f = (
        brax_eval_n_times if config["evo"]["n_evaluations"] > 1 else brax_eval
    )

    env = get_braxv1_env(config)
    partial_eval_f = partial(evaluation_f, decoder=decoder, config=config, env=env)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    overall_best_member = {"individual": state.mean, "fitness": 0}

    for _generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)

        # NOTE - Ask
        x, state = ask(rng_gen, state)

        # NOTE - Eval
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        state = tell(x, fitness, state)

        # NOTE - update best
        all_members = jnp.vstack((x, overall_best_member["individual"]))
        all_fitnesses = jnp.hstack((true_fitness, overall_best_member["fitness"]))
        best_member_i = jnp.argmax(all_fitnesses)
        # Overwrite best member
        overall_best_member = {
            "individual": all_members[best_member_i],
            "fitness": all_fitnesses[best_member_i],
        }

    return overall_best_member["fitness"]
    # return partial_eval_f(state.mean, rng_eval)


def learn_brax_task_untracked_nn_df_d0_50(
    df_genotype: Array,
    rng: jrd.KeyArray,
    meta_decoder: Decoder,
    df_model: nn.Module,
    config: dict,
) -> float:
    """Run an es training loop specifically tailored for brax tasks.

    Args:
        config (dict): config of the current run.
        df (DistanceFunction): Distance function to use, can be parametrized.
        wdb_run: wandb run object used to log

    Returns:
        float: fitness of the overall best individual encountered
    """
    rng, rng_init = jrd.split(rng, 2)

    # NOTE - The distance function genotype needs to be decoded here
    # because objects cannot be given as input of vectorized functions
    df_phenotype = meta_decoder.decode(df_genotype)
    df = NNDistanceSimple(model_parameters=df_phenotype, model=df_model)
    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=decoder.encoding_size(),
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    # Each individual is evaluated a single time or multiple times in parallel
    evaluation_f = (
        brax_eval_n_times if config["evo"]["n_evaluations"] > 1 else brax_eval
    )

    env = get_braxv1_env(config)
    partial_eval_f = partial(evaluation_f, decoder=decoder, config=config, env=env)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    initial_mean_fitness: float = 0.0
    final_mean_fitness: float = 0.0

    for _generation in range(50):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)

        # NOTE - metrics
        if _generation == 0:
            initial_mean_fitness = partial_eval_f(state.mean, rng_eval)
        # NOTE - Ask
        x, state = ask(rng_gen, state)

        # NOTE - Eval
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        state = tell(x, fitness, state)

        # NOTE - metrics
        if _generation == range(50)[-1]:
            final_mean_fitness = partial_eval_f(state.mean, rng_eval)

    return final_mean_fitness - initial_mean_fitness


# ================================================
# ============   CGP Learning   ==================
# ================================================


def learn_brax_task_cgp_d0_50(
    cgp_genome: Array,
    rng: jrd.KeyArray,
    config: dict,
    cgp_config: dict,
) -> float:
    """Run an es training loop specifically tailored for brax tasks
    using a cgp defined function.

        Returns:
        float: fitness of the overall best individual encountered
    """
    rng, rng_init = jrd.split(rng, 2)

    df = CGPDistance(cgp_genome=cgp_genome, cgp_config=cgp_config)
    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=decoder.encoding_size(),
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    # Each individual is evaluated a single time or multiple times in parallel
    evaluation_f = (
        brax_eval_n_times if config["evo"]["n_evaluations"] > 1 else brax_eval
    )

    env = get_braxv1_env(config)
    partial_eval_f = partial(evaluation_f, decoder=decoder, config=config, env=env)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    initial_mean_fitness: float = 0.0
    final_mean_fitness: float = 0.0
    for _generation in range(50):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)

        # NOTE - metrics
        if _generation == 0:
            initial_mean_fitness = partial_eval_f(state.mean, rng_eval)

        # NOTE - Ask
        x, state = ask(rng_gen, state)

        # NOTE - Eval
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        state = tell(x, fitness, state)

        # NOTE - metrics
        if _generation == range(50)[-1]:
            final_mean_fitness = partial_eval_f(state.mean, rng_eval)

    return final_mean_fitness - initial_mean_fitness


# ============================================================
# =====================  Gymnax  =============================
# ============================================================
def gymnax_eval(
    genome: Array,
    rng: jrd.KeyArray,
    decoder: Decoder,
    config: dict,
) -> float:
    model_parameters = decoder.decode(genome)
    model = Models[config["net"]["architecture"]](config)

    return rollout_gymnax_task(model, model_parameters, rng, config)


def learn_gymnax_task(
    df: DistanceFunction, rng: jrd.KeyArray, config: dict, wandb_run
) -> float:
    """Runs a gymnax learning loop defined by a config file.
    The distance function has to be decoded from its `df_genotype`.
    """
    rng, rng_init = jrd.split(rng, 2)

    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"], num_dims=decoder.encoding_size()
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    partial_eval_f = partial(gymnax_eval, decoder=decoder, config=config)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    tracker = Tracker(config, decoder)
    tracker_state = tracker.init(True)

    for _generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = ask(rng_gen, state)
        # NOTE - Evaluate
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1.0 * true_fitness if config["task"]["maximize"] else true_fitness
        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = tell(x, fitness, state)

        # NOTE - stats
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=x,
            fitnesses=true_fitness,
            sample_mean=state.mean,
            eval_f=partial_eval_f,
            rng_eval=rng_eval,
            skip_mean_backup=True,
        )
        tracker.wandb_log(tracker_state, wandb_run)

    return tracker.get_top_k_genomes(tracker_state)[0]


def learn_gymnax_task_nn_df(
    df_genotype: Array,
    rng: jrd.KeyArray,
    meta_decoder: Decoder,
    df_model: nn.Module,
    config: dict,
) -> float:
    """Runs a gymnax learning loop using GENE encoding with a neural network
    based distance function.
    The distance function has to be decoded from its `df_genotype`.

    Args:
        df_genotype (Array): The genotype of the distance function.
        rng (jrd.KeyArray): rng key used for initialization and training.
        meta_decoder (Decoder): Decoder used to decode the distance funcion genotypes.
        model (nn.Module): the model used as the distance function.
        config (dict): config file used to specify the current runs values.

    Returns:
        float: fitness of the overall best individual encountered
    """
    rng, rng_init = jrd.split(rng, 2)

    # NOTE - The distance function genotype needs to be decoded here
    # because objects cannot be given as input of vectorized functions
    df_phenotype = meta_decoder.decode(df_genotype)
    df = NNDistanceSimple(model_parameters=df_phenotype, model=df_model)
    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"], num_dims=decoder.encoding_size()
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    partial_eval_f = partial(gymnax_eval, decoder=decoder, config=config)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    overall_best_member = {"individual": state.mean, "fitness": 0}

    for _generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = ask(rng_gen, state)
        # NOTE - Evaluate
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1.0 * true_fitness if config["task"]["maximize"] else true_fitness
        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = tell(x, fitness, state)

        # NOTE - update best
        all_members = jnp.vstack((x, overall_best_member["individual"]))
        all_fitnesses = jnp.hstack((true_fitness, overall_best_member["fitness"]))
        best_member_i = jnp.argmax(all_fitnesses)
        # Overwrite best member
        overall_best_member = {
            "individual": all_members[best_member_i],
            "fitness": all_fitnesses[best_member_i],
        }

    return overall_best_member["fitness"]
    # return partial_eval_f(state.mean, rng_eval)


def learn_gymnax_task_cgp_df_max(
    cgp_genome: Array,
    rng: jrd.KeyArray,
    config: dict,
    cgp_config: dict,
) -> float:
    """Runs a gymnax learning loop using GENE encoding with a CGP
    based distance function.
    The distance function has to be decoded from its `cgp_genome`.

    Returns:
        float: fitness of the overall best individual encountered
    """
    rng, rng_init = jrd.split(rng, 2)

    df = CGPDistance(cgp_genome, cgp_config)
    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"], num_dims=decoder.encoding_size()
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    partial_eval_f = partial(gymnax_eval, decoder=decoder, config=config)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    overall_best_member = {"individual": state.mean, "fitness": 0}

    for _generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = ask(rng_gen, state)
        # NOTE - Evaluate
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1.0 * true_fitness if config["task"]["maximize"] else true_fitness
        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = tell(x, fitness, state)

        # NOTE - update best
        all_members = jnp.vstack((x, overall_best_member["individual"]))
        all_fitnesses = jnp.hstack((true_fitness, overall_best_member["fitness"]))
        best_member_i = jnp.argmax(all_fitnesses)
        # Overwrite best member
        overall_best_member = {
            "individual": all_members[best_member_i],
            "fitness": all_fitnesses[best_member_i],
        }

    return overall_best_member["fitness"]
    # return partial_eval_f(state.mean, rng_eval)


def learn_gymnax_task_cgp_df_mean(
    cgp_genome: Array,
    rng: jrd.KeyArray,
    config: dict,
    cgp_config: dict,
) -> float:
    """Runs a gymnax learning loop using GENE encoding with a CGP
    based distance function.
    The distance function has to be decoded from its `cgp_genome`.

    Returns:
        float: fitness of the overall best individual encountered
    """
    rng, rng_init = jrd.split(rng, 2)

    df = CGPDistance(cgp_genome, cgp_config)
    decoder = get_decoder(config)(config, df)

    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"], num_dims=decoder.encoding_size()
    )
    state = strategy.initialize(rng_init)
    ask = jit(strategy.ask)
    tell = jit(strategy.tell)

    partial_eval_f = partial(gymnax_eval, decoder=decoder, config=config)
    vectorized_eval_f = jit(vmap(partial_eval_f, in_axes=(0, None)))

    for _generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = ask(rng_gen, state)
        # NOTE - Evaluate
        true_fitness = vectorized_eval_f(x, rng_eval)
        fitness = -1.0 * true_fitness if config["task"]["maximize"] else true_fitness
        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = tell(x, fitness, state)

    return partial_eval_f(state.mean, rng_eval)
