from functools import partial

import evosax
import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap

from gene.core.decoding import DirectDecoder
from gene.core.models import ReluLinearModel
from gene.learning import (
    learn_gymnax_task,
    learn_brax_task_untracked,
    learn_gymnax_task_cgp_df,
)
from gene.tracker import MetaTracker

from cgpax.jax_individual import (
    generate_population,
    compute_cgp_genome_mask,
    compute_cgp_mutation_prob_mask,
    mutate_genome_n_times,
)
from cgpax.jax_selection import fp_selection
from cgpax.utils import readable_cgp_program_from_genome


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
    nn_dst_model = ReluLinearModel(config["net"]["layer_dimensions"][1:])

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
        _norm_f_cp = f_cp / (max_f_cp if max_f_cp != 0 else 1.0)
        _norm_f_hc_100 = f_hc_100 / (max_f_hc_100 if max_f_hc_100 != 0 else 1.0)
        _norm_f_hc_1000 = f_hc_1000 / (max_f_hc_1000 if max_f_hc_1000 != 0 else 1.0)

        # Fitness used to inform the strategy update (tell)
        # true_fitness = f_cp + f_hc_100 + f_hc_1000
        true_fitness = _norm_f_cp + _norm_f_hc_100 + _norm_f_hc_1000
        # true_fitness = _norm_f_cp + 10 * _norm_f_hc_100 + 10e1 * _norm_f_hc_1000
        fitness = -1 * true_fitness if config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        meta_state = tell(x, fitness, meta_state)

        # NOTE - Tracker
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness_value={
                "total_emp_mean_fitness": jnp.mean(true_fitness),
                "total_max_fitness": jnp.max(true_fitness),
                "cart": {
                    "max_fitness": max_f_cp,
                    "emp_mean_fitnesses": jnp.mean(f_cp),
                },
                "hc_100": {
                    "max_fitness": max_f_hc_100,
                    "emp_mean_fitnesses": jnp.mean(f_hc_100),
                },
                "hc_1000": {
                    "max_fitness": max_f_hc_1000,
                    "emp_mean_fitnesses": jnp.mean(f_hc_1000),
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


def meta_learn_cgp(meta_config: dict, cgp_config: dict):
    """Meta evolution of a cgp parametrized distance function"""
    assert cgp_config["n_individuals"] == meta_config["evo"]["population_size"]

    rng = jrd.PRNGKey(meta_config["seed"])

    # Evaluation function based on CGP using CGP df
    # Input size is the number of values for each neuron position vector
    # Output size is 1, the distance between the two neurons
    cgp_config["n_in_env"] = meta_config["encoding"]["d"] * 2
    #  hard to create constants, so hardcode them in as input
    cgp_config["n_constants"] = 1
    cgp_config["n_in"] = cgp_config["n_in_env"] = cgp_config["n_constants"]
    cgp_config["n_out"] = 1

    cgp_config["buffer_size"] = cgp_config["n_in"] + cgp_config["n_nodes"]
    cgp_config["genome_size"] = 3 * cgp_config["n_nodes"] + cgp_config["n_out"]
    n_mutations_per_individual = int(
        (cgp_config["n_individuals"] - cgp_config["elite_size"])
        / cgp_config["elite_size"]
    )
    nan_replacement = cgp_config["nan_replacement"]

    # preliminary evo steps
    genome_mask = compute_cgp_genome_mask(
        cgp_config, n_in=cgp_config["n_in"], n_out=cgp_config["n_out"]
    )
    mutation_mask = compute_cgp_mutation_prob_mask(
        cgp_config, n_out=cgp_config["n_out"]
    )

    # evaluation
    vec_learn_cartpole = jit(
        vmap(
            partial(
                learn_gymnax_task_cgp_df,
                config=meta_config["curriculum"]["cart"],
                cgp_config=cgp_config,
            ),
            in_axes=(0, None),
        )
    )

    partial_fp_selection = partial(fp_selection, n_elites=cgp_config["elite_size"])
    jit_partial_fp_selection = jit(partial_fp_selection)
    # mutation
    partial_multiple_mutations = partial(
        mutate_genome_n_times,
        n_mutations=n_mutations_per_individual,
        genome_mask=genome_mask,
        mutation_mask=mutation_mask,
    )
    vmap_multiple_mutations = vmap(partial_multiple_mutations)
    jit_vmap_multiple_mutations = jit(vmap_multiple_mutations)
    # replace invalid fitness values
    fitness_replacement = jit(partial(jnp.nan_to_num, nan=nan_replacement))

    rng, rng_generation = jrd.split(rng, 2)
    genomes = generate_population(
        pop_size=cgp_config["n_individuals"],
        genome_mask=genome_mask,
        rnd_key=rng_generation,
    )

    for _meta_generation in range(meta_config["evo"]["n_generations"]):
        rng, rng_eval = jrd.split(rng, 2)
        # NOTE - evaluate population
        fitness_values = vec_learn_cartpole(genomes, rng_eval)
        fitness_values = fitness_replacement(fitness_values)
        # fitness_values = -fitness_values  # we would need to minimize the error

        # NOTE - select parents
        rng, rng_fp = jrd.split(rng, 2)
        # Choose selection mechanism
        parents = jit_partial_fp_selection(genomes, fitness_values, rng_fp)

        # NOTE - compute offspring
        rng, rng_mutation = jrd.split(rng, 2)
        # FIXME - fix this
        mutate_keys = jrd.split(rng_mutation, len(parents))
        new_genomes_matrix = jit_vmap_multiple_mutations(parents, mutate_keys)
        new_genomes = jnp.reshape(
            new_genomes_matrix, (-1, new_genomes_matrix.shape[-1])
        )

        # max index
        best_genome = genomes.at[jnp.argmax(fitness_values)].get()
        best_fitness = jnp.max(fitness_values)
        best_program = readable_cgp_program_from_genome(best_genome, cgp_config)

        # print progress
        print(f"[Meta gen {_meta_generation}] - best fitness: {best_fitness}")
        print(best_program)

        # NOTE - update population
        genomes = jnp.concatenate((parents, new_genomes))

    return None
