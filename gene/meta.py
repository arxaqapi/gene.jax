from functools import partial
from pathlib import Path

import evosax
import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap

from gene.core.decoding import DirectDecoder
from gene.core.models import ReluLinearModel
from gene.learning import (
    learn_gymnax_task_nn_df,
    learn_brax_task_cgp,
    learn_brax_task_untracked_nn_df,
    learn_gymnax_task_cgp_df_mean,
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
from cgpax.run_utils import (
    __update_config_with_data__,
    __compute_masks__,
    __compile_mutation__,
    __compute_genome_transformation_function__,
)
from cgpax.analysis.genome_analysis import __save_graph__, __write_readable_program__


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
                learn_gymnax_task_nn_df,
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
                learn_brax_task_untracked_nn_df,
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
                learn_brax_task_untracked_nn_df,
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

        f_hc_100 = vec_learn_hc_100(x, rng_eval) if max_f_cp > 400 else 0
        max_f_hc_100 = jnp.max(f_hc_100)

        f_hc_1000 = vec_learn_hc_1000(x, rng_eval) if max_f_hc_100 > 200 else 0
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


def meta_learn_cgp(meta_config: dict, cgp_config: dict, wandb_run=None):
    """Meta evolution of a cgp parametrized distance function"""
    assert cgp_config["n_individuals"] == meta_config["evo"]["population_size"]

    rng: jrd.KeyArray = jrd.PRNGKey(meta_config["seed"])

    # Evaluation function based on CGP using CGP df
    # Input size is the number of values for each neuron position vector
    # Output size is 1, the distance between the two neurons
    __update_config_with_data__(
        cgp_config,
        observation_space_size=meta_config["encoding"]["d"] * 2,
        action_space_size=1,
    )
    # n_mutations_per_individual = int(
    #     (cgp_config["n_individuals"] - cgp_config["elite_size"])
    #     / cgp_config["elite_size"]
    # )
    nan_replacement = cgp_config["nan_replacement"]

    # preliminary evo steps
    genome_mask, mutation_mask = __compute_masks__(cgp_config)

    # evaluation curriculum fonctions
    # NOTE - removed JIT
    vec_learn_hc_100 = vmap(
        partial(
            learn_brax_task_cgp,
            config=meta_config["curriculum"]["hc_100"],
            cgp_config=cgp_config,
        ),
        in_axes=(0, None),
    )
    # NOTE - removed JIT
    vec_learn_hc_500 = vmap(
        partial(
            learn_brax_task_cgp,
            config=meta_config["curriculum"]["hc_500"],
            cgp_config=cgp_config,
        ),
        in_axes=(0, None),
    )

    partial_fp_selection = partial(fp_selection, n_elites=cgp_config["elite_size"])
    jit_partial_fp_selection = jit(partial_fp_selection)
    # mutation
    genome_transformation_function = __compute_genome_transformation_function__(
        cgp_config
    )
    batch_mutate_genomes = __compile_mutation__(
        cgp_config,
        genome_mask,
        mutation_mask,
        genome_transformation_function=genome_transformation_function,
    )

    # replace invalid fitness values
    fitness_nan_replacement = jit(partial(jnp.nan_to_num, nan=nan_replacement))

    rng, rng_generation = jrd.split(rng, 2)
    genomes = generate_population(
        pop_size=cgp_config["n_individuals"],
        genome_mask=genome_mask,
        rnd_key=rng_generation,
        genome_transformation_function=genome_transformation_function,
    )

    wandb_run.config.update(meta_config, allow_val_change=True)
    for _meta_generation in range(meta_config["evo"]["n_generations"]):
        print(f"[Meta gen {_meta_generation}] - Start")
        rng, rng_eval = jrd.split(rng, 2)
        # NOTE - evaluate population on curriculum of tasks
        f_hc_100 = vec_learn_hc_100(genomes, rng_eval)
        print(f"[Meta gen {_meta_generation}] - eval hc 100 done")
        f_hc_500 = vec_learn_hc_500(genomes, rng_eval)
        print(f"[Meta gen {_meta_generation}] - eval hc 500 done")
        fitness_values = f_hc_100 + f_hc_500

        # NAN replacement
        fitness_values = fitness_nan_replacement(fitness_values)

        # NOTE - select parents
        rng, rng_fp = jrd.split(rng, 2)
        # Choose selection mechanism
        parents = jit_partial_fp_selection(genomes, fitness_values, rng_fp)

        # NOTE - compute offspring
        rng, rng_mutation = jrd.split(rng, 2)
        rng_multiple_mutations = jrd.split(rng_mutation, len(parents))
        new_genomes_matrix = batch_mutate_genomes(parents, rng_multiple_mutations)
        new_genomes = jnp.reshape(
            new_genomes_matrix, (-1, new_genomes_matrix.shape[-1])
        )

        # max index
        best_genome_idx = jnp.argmax(fitness_values)
        best_genome = genomes[best_genome_idx]
        best_fitness = fitness_values[best_genome_idx]
        best_program = readable_cgp_program_from_genome(best_genome, cgp_config)

        # print progress
        print(f"[Meta gen {_meta_generation}] - best fitness: {best_fitness}")
        print(best_program)

        # NOTE - update population
        genomes = jnp.concatenate((parents, new_genomes))

        # NOTE - log stats
        if wandb_run is not None:
            wandb_run.log(
                {
                    "training": {
                        "total_emp_mean_fitness": fitness_values.mean(),
                        "total_max_fitness": fitness_values.max(),
                        "hc100": {
                            "emp_mean_fit": f_hc_100.mean(),
                            "max_fit": f_hc_100.max(),
                        },
                        "hc500": {
                            "emp_mean_fit": f_hc_500.mean(),
                            "max_fit": f_hc_500.max(),
                        },
                    },
                }
            )
            # Save best genome as graph and readable program
            programm_save_path = Path(wandb_run.dir) / "programs"
            programm_save_path.mkdir(parents=True, exist_ok=True)
            graph_save_path = str(
                programm_save_path / f"gen_{_meta_generation}_best_graph.png"
            )
            readable_programm_save_path = str(
                programm_save_path / f"gen_{_meta_generation}_best_programm.txt"
            )
            __save_graph__(
                genome=best_genome,
                config=cgp_config,
                file=graph_save_path,
                input_color="green",
                output_color="red",
            )
            __write_readable_program__(
                genome=best_genome,
                config=cgp_config,
                target_file=readable_programm_save_path,
            )
            # Save best
            save_path = (
                Path(wandb_run.dir)
                / "df_genomes"
                / f"mg_{_meta_generation}_best_genome.npy"
            )
            save_path = save_path.with_suffix(".npy")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                jnp.save(f, best_genome)

            wandb_run.save(
                str(graph_save_path), base_path=f"{wandb_run.dir}/", policy="now"
            )
            wandb_run.save(
                str(readable_programm_save_path),
                base_path=f"{wandb_run.dir}/",
                policy="now",
            )
            wandb_run.save(str(save_path), base_path=f"{wandb_run.dir}/", policy="now")

        print(f"[Meta gen {_meta_generation}] - End\n")

    return None


def meta_learn_cgp_simple(meta_config: dict, cgp_config: dict, wandb_run=None):
    """Gymnax-only environnments meta-learning (cartpole, Acrobot) for a
    cgp parametrized distance function
    """
    rng = jrd.PRNGKey(meta_config["seed"])

    # Evaluation function based on CGP using CGP df
    # Input size is the number of values for each neuron position vector
    # Output size is 1, the distance between the two neurons
    __update_config_with_data__(
        cgp_config,
        observation_space_size=meta_config["encoding"]["d"] * 2,
        action_space_size=1,
    )
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
                learn_gymnax_task_cgp_df_mean,
                config=meta_config["curriculum"]["cart"],
                cgp_config=cgp_config,
            ),
            in_axes=(0, None),
        )
    )
    vec_learn_acrobot = jit(
        vmap(
            partial(
                learn_gymnax_task_cgp_df_mean,
                config=meta_config["curriculum"]["acrobot"],
                cgp_config=cgp_config,
            ),
            in_axes=(0, None),
        )
    )

    # fp select
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
        print(f"[Meta gen {_meta_generation}] - Start")
        rng, rng_eval = jrd.split(rng, 2)
        # NOTE - evaluate population on curriculum
        f_cartpole = vec_learn_cartpole(genomes, rng_eval)
        max_f_cartpole = jnp.max(f_cartpole)

        f_acrobot = vec_learn_acrobot(genomes, rng_eval) if max_f_cartpole > 400 else 0
        max_f_acrobot = jnp.max(f_acrobot)

        fitness_values = f_cartpole + f_acrobot
        # fix nan values
        fitness_values = fitness_replacement(fitness_values)

        # NOTE - select parents
        rng, rng_fp = jrd.split(rng, 2)
        # Choose selection mechanism
        parents = jit_partial_fp_selection(genomes, fitness_values, rng_fp)

        # NOTE - compute offspring
        rng, rng_mutation = jrd.split(rng, 2)
        rng_multiple_mutations = jrd.split(rng_mutation, len(parents))
        new_genomes_matrix = jit_vmap_multiple_mutations(
            parents, rng_multiple_mutations
        )
        new_genomes = jnp.reshape(
            new_genomes_matrix, (-1, new_genomes_matrix.shape[-1])
        )

        # max index
        best_genome_idx = jnp.argmax(fitness_values)
        best_genome = genomes[best_genome_idx]
        best_fitness = fitness_values[best_genome_idx]
        best_program = readable_cgp_program_from_genome(best_genome, cgp_config)

        # print progress
        print(f"[Meta gen {_meta_generation}] - best fitness: {best_fitness}")
        print(f"\t{max_f_cartpole}")
        print(f"\t{max_f_acrobot}")
        print(best_program)

        if wandb_run is not None:
            wandb_run.log({"best_fitness": best_fitness})

        # NOTE - update population
        genomes = jnp.concatenate((parents, new_genomes))
        print(f"[Meta gen {_meta_generation}] - End\n")

    return None
