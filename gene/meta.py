from functools import partial

import evosax
import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap

from gene.core.decoding import DirectDecoder
from gene.core.models import LinearModel
from gene.learning import learn_gymnax_task, learn_brax_task_untracked
from gene.tracker import MetaTracker


def meta_learn_nn(config: dict, wandb_run):
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
    vec_learn_hc_100 = jit(
        vmap(
            partial(
                learn_brax_task_untracked,
                meta_decoder=meta_decoder,
                df_model=nn_dst_model,
                config=config["curriculum"]["hc_100"],
            ),
            in_axes=(0, None),
        )
    )
    vec_learn_hc_1000 = jit(
        vmap(
            partial(
                learn_brax_task_untracked,
                meta_decoder=meta_decoder,
                df_model=nn_dst_model,
                config=config["curriculum"]["hc_1000"],
            ),
            in_axes=(0, None),
        )
    )

    tracker = MetaTracker(config, meta_decoder)
    tracker_state = tracker.init()

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
        # All distance functions (x) are evaluated by running a complete policy learning
        # loop using GENE with a nn distance function, the sample mean is then evaluated
        f_cp = vec_learn_cartpole(x, rng_eval)
        max_f_cp = jnp.max(f_cp)

        f_hc_100 = (
            vec_learn_hc_100(x, rng_eval)
            if max_f_cp > 400
            else jnp.zeros((config["evo"]["population_size"],))
        )
        max_f_hc_100 = jnp.max(f_hc_100)

        f_hc_1000 = (
            vec_learn_hc_1000(x, rng_eval)
            if max_f_hc_100 > 200
            else jnp.zeros((config["evo"]["population_size"],))
        )
        max_f_hc_1000 = jnp.max(f_hc_1000)
        # NOTE - 3. aggregate fitnesses and weight them
        norm_f_cp = f_cp / (max_f_cp if max_f_cp != 0 else 1.0)
        norm_f_hc_100 = f_hc_100 / (max_f_hc_100 if max_f_hc_100 != 0 else 1.0)
        norm_f_hc_1000 = f_hc_1000 / (max_f_hc_1000 if max_f_hc_1000 != 0 else 1.0)

        true_fitness = norm_f_cp + 10 * norm_f_hc_100 + 10e1 * norm_f_hc_1000
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness
        print(f"{true_fitness=}")

        # NOTE - Tell
        meta_state = tell(x, fitness, meta_state)

        # NOTE - Tracker
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness_value={
                "total_emp_mean_fitness": jnp.mean(true_fitness),
                "cart": {
                    "max_fitness": max_f_cp,
                    "emp_mean_fitnesses": jnp.mean(f_cp),
                },
                "hc_100": {
                    "max_fitness": max_f_hc_100,
                    "emp_mean_fitnesses": jnp.mean(max_f_hc_100),
                },
                "hc_1000": {
                    "max_fitness": max_f_hc_1000,
                    "emp_mean_fitnesses": jnp.mean(max_f_hc_1000),
                },
            },
            max_df=x[jnp.argmax(true_fitness)],
            mean_df=meta_state.mean,
            gen=meta_generation,
        )
        if wandb_run is not None:
            tracker.wandb_log(tracker_state, wandb_run)
            tracker.wandb_save_genome(
                meta_state.mean,
                wandb_run,
                file_name=f"g{str(meta_generation).zfill(3)}_mean_df_indiv",
                now=True,
            )

    return meta_state.mean


# def meta_learn_cgp(config: dict):
#     """Meta evolution of a cgp parametrized distance function"""
#     raise NotImplementedError
