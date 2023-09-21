from functools import partial
from pathlib import Path

import evosax
import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap

from gene.core.decoding import DirectDecoder
from gene.core.models import ReluLinearModel, get_model
from gene.core.evaluation import evaluate_rand_network_properties_n_times
from gene.learning import (
    learn_gymnax_task_nn_df,
    learn_brax_task_cgp,
    learn_brax_task_untracked_nn_df,
    learn_brax_task_untracked_nn_df_d0_50,
    # learn_gymnax_task_cgp_df_mean,
    learn_brax_task_cgp_d0_50,
)
from gene.utils import min_max_scaler
from gene.tracker import MetaTracker

from cgpax.jax_individual import generate_population
from cgpax.jax_encoding import genome_to_cgp_program
from cgpax.utils import readable_cgp_program_from_genome, compute_active_size
from cgpax.run_utils import (
    __update_config_with_data__,
    __compute_masks__,
    __compile_mutation__,
    __compute_genome_transformation_function__,
    __compile_parents_selection__,
    __compile_survival_selection__,
)
from cgpax.analysis.genome_analysis import __save_graph__, __write_readable_program__


# ================================================
# ==============   NN Dist   =====================
# ================================================
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
        rng, rng_gen, rng_eval_cp, rng_eval_hc_100, rng_eval_hc_1000 = jrd.split(rng, 5)
        # NOTE - Ask
        x, meta_state = ask(rng_gen, meta_state)

        # NOTE - Evaluation curriculum
        # NOTE - 1. DF genome to DF using decoder and NNDF, array of df
        # Cannot create an array of DF and pass it down to vectorized functions because
        # an array cannot contain objects (only floats and bools)
        # NOTE - 2. Complete training and evaluation on a curriculum of tasks
        # All distance functions (x) are evaluated by running a complete policy learning
        # loop using GENE with a nn distance function, the sample mean is then evaluated
        f_cp = vec_learn_cartpole(x, rng_eval_cp)
        max_f_cp = jnp.max(f_cp)

        f_hc_100 = vec_learn_hc_100(x, rng_eval_hc_100) if max_f_cp > 400 else 0
        max_f_hc_100 = jnp.max(f_hc_100)

        f_hc_1000 = vec_learn_hc_1000(x, rng_eval_hc_1000) if max_f_hc_100 > 200 else 0
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


def meta_learn_nn_corrected(meta_config: dict, wandb_run, beta: float = 0.5):
    rng = jrd.PRNGKey(meta_config["seed"])
    rng, rng_init = jrd.split(rng, 2)

    meta_decoder = DirectDecoder(meta_config)

    # meta_strategy = evosax.Sep_CMA_ES(
    meta_strategy = evosax.DES(
        popsize=meta_config["evo"]["population_size"],
        num_dims=meta_decoder.encoding_size(),
    )
    meta_state = meta_strategy.initialize(rng_init)

    ask = jit(meta_strategy.ask)
    tell = jit(meta_strategy.tell)

    # Neural network distance network
    assert meta_config["net"]["architecture"] in [
        "relu_linear",
        "tanh_linear",
        "relu_tanh_linear",
    ]
    nn_df_model = get_model(meta_config)

    if beta < 1:
        # NOTE - JIT removed
        vec_learn_hc_500 = vmap(
            partial(
                learn_brax_task_untracked_nn_df,
                meta_decoder=meta_decoder,
                df_model=nn_df_model,
                config=meta_config["curriculum"]["hc_500"],
            ),
            in_axes=(0, None),
        )
        vec_learn_brax_task_untracked_nn_df_d0_50 = vmap(
            partial(
                learn_brax_task_untracked_nn_df_d0_50,
                meta_decoder=meta_decoder,
                df_model=nn_df_model,
                config=meta_config["curriculum"]["hc_500"],
            ),
            in_axes=(0, None),
        )

    vec_evaluate_network_properties_nn_dist = jit(
        vmap(
            partial(
                evaluate_rand_network_properties_n_times,
                meta_config=meta_config,
                df_type="nn",
                n=32,
            ),
            in_axes=(0, 0),
        )
    )

    for meta_generation in range(meta_config["evo"]["n_generations"]):
        print(f"[Meta gen n°{meta_generation:>5}]")
        rng, rng_gen, rng_eval_hc_500, rng_eval_net_prop = jrd.split(rng, 4)
        rng_eval_net_prop = jrd.split(
            rng_eval_net_prop, meta_config["evo"]["population_size"]
        )
        # NOTE - Ask
        x, meta_state = ask(rng_gen, meta_state)

        if beta < 1:
            # NOTE - Evaluation curriculum
            f_hc_500 = vec_learn_hc_500(x, rng_eval_hc_500)
            f_hc_500_d0_50 = vec_learn_brax_task_untracked_nn_df_d0_50(
                x, rng_eval_hc_500
            )
            f_policy_eval = min_max_scaler(f_hc_500) + min_max_scaler(f_hc_500_d0_50)
        else:
            f_policy_eval = 0

        f_expr, f_w_distr, f_inp = vec_evaluate_network_properties_nn_dist(
            x, rng_eval_net_prop
        )
        f_net_prop = (
            min_max_scaler(f_expr) + min_max_scaler(f_w_distr) + min_max_scaler(f_inp)
        )

        true_fitness = beta * f_net_prop + (1 - beta) * f_policy_eval
        assert true_fitness is not None
        fitness = -1 * true_fitness if meta_config["task"]["maximize"] else true_fitness

        # NOTE - Tell
        meta_state = tell(x, fitness, meta_state)

        # NOTE - Tracker
        if wandb_run is not None:
            to_log = {
                "training": {
                    "total_emp_mean_fitness": true_fitness.mean(),
                    "total_max_fitness": true_fitness.max(),
                    "net_prop": {
                        "f_expressivity": f_expr.mean(),
                        "f_weight_distribution": f_w_distr.mean(),
                        "f_input_restoration": f_inp.mean(),
                    },
                },
            }
            if beta < 1:
                to_log["training"]["hc_500"] = {
                    "max_fitness": f_hc_500.max(),
                    "emp_mean_fitnesses": f_hc_500.mean(),
                }
                to_log["training"]["hc500_d0_50"] = {
                    "emp_mean_fit": f_hc_500_d0_50.mean(),
                    "max": f_hc_500_d0_50.max(),
                }
            wandb_run.log(to_log)
            # Save best
            best_genome = x[jnp.argmax(true_fitness)]
            save_path = (
                Path(wandb_run.dir)
                / "df_genomes"
                / f"mg_{meta_generation}_best_genome.npy"
            )
            save_path = save_path.with_suffix(".npy")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                jnp.save(f, best_genome)
            wandb_run.save(str(save_path), base_path=f"{wandb_run.dir}/", policy="now")

    return meta_state.mean, best_genome, nn_df_model


# ================================================
# ==============   CGP Dist   ====================
# ================================================


# def meta_learn_cgp(meta_config: dict, wandb_run=None):
#     """Meta evolution of a cgp parametrized distance function"""
#     assert (
#         meta_config["cgp_config"]["n_individuals"]
#         == meta_config["evo"]["population_size"]
#     )

#     rng: jrd.KeyArray = jrd.PRNGKey(meta_config["seed"])

#     # Evaluation function based on CGP using CGP df
#     # Input size is the number of values for each neuron position vector
#     # Output size is 1, the distance between the two neurons
#     __update_config_with_data__(
#         meta_config["cgp_config"],
#         observation_space_size=meta_config["encoding"]["d"] * 2,
#         action_space_size=1,
#     )
#     # n_mutations_per_individual = int(
#     #     (cgp_config["n_individuals"] - cgp_config["elite_size"])
#     #     / cgp_config["elite_size"]
#     # )
#     nan_replacement = meta_config["cgp_config"]["nan_replacement"]

#     # preliminary evo steps
#     genome_mask, mutation_mask = __compute_masks__(meta_config["cgp_config"])

#     # evaluation curriculum fonctions
#     # NOTE - removed JIT
#     vec_learn_hc_100 = vmap(
#         partial(
#             learn_brax_task_cgp,
#             config=meta_config["curriculum"]["hc_100"],
#             cgp_config=meta_config["cgp_config"],
#         ),
#         in_axes=(0, None),
#     )
#     # NOTE - removed JIT
#     vec_learn_hc_500 = vmap(
#         partial(
#             learn_brax_task_cgp,
#             config=meta_config["curriculum"]["hc_500"],
#             cgp_config=meta_config["cgp_config"],
#         ),
#         in_axes=(0, None),
#     )

#     partial_fp_selection = partial(
#         fp_selection, n_elites=meta_config["cgp_config"]["elite_size"]
#     )
#     jit_partial_fp_selection = jit(partial_fp_selection)
#     # mutation
#     genome_transformation_function = __compute_genome_transformation_function__(
#         meta_config["cgp_config"]
#     )
#     batch_mutate_genomes = __compile_mutation__(
#         meta_config["cgp_config"],
#         genome_mask,
#         mutation_mask,
#         genome_transformation_function=genome_transformation_function,
#     )

#     # replace invalid fitness values
#     fitness_nan_replacement = jit(partial(jnp.nan_to_num, nan=nan_replacement))

#     rng, rng_generation = jrd.split(rng, 2)
#     genomes = generate_population(
#         pop_size=meta_config["cgp_config"]["n_individuals"],
#         genome_mask=genome_mask,
#         rnd_key=rng_generation,
#         genome_transformation_function=genome_transformation_function,
#     )

#     wandb_run.config.update(meta_config, allow_val_change=True)
#     for _meta_generation in range(meta_config["evo"]["n_generations"]):
#         print(f"[Meta gen {_meta_generation}] - Start")
#         rng, rng_eval_hc100, rng_eval_hc500 = jrd.split(rng, 3)
#         # NOTE - evaluate population on curriculum of tasks
#         f_hc_100 = vec_learn_hc_100(genomes, rng_eval_hc100)
#         print(f"[Meta gen {_meta_generation}] - eval hc 100 done")
#         f_hc_500 = vec_learn_hc_500(genomes, rng_eval_hc500)
#         print(f"[Meta gen {_meta_generation}] - eval hc 500 done")
#         fitness_values = f_hc_100 + f_hc_500

#         # NAN replacement
#         fitness_values = fitness_nan_replacement(fitness_values)

#         # NOTE - select parents
#         rng, rng_fp = jrd.split(rng, 2)
#         # Choose selection mechanism
#         parents = jit_partial_fp_selection(genomes, fitness_values, rng_fp)

#         # NOTE - compute offspring
#         rng, rng_mutation = jrd.split(rng, 2)
#         rng_multiple_mutations = jrd.split(rng_mutation, len(parents))
#         new_genomes_matrix = batch_mutate_genomes(parents, rng_multiple_mutations)
#         new_genomes = jnp.reshape(
#             new_genomes_matrix, (-1, new_genomes_matrix.shape[-1])
#         )

#         # max index
#         best_genome_idx = jnp.argmax(fitness_values)
#         best_genome = genomes[best_genome_idx]
#         best_fitness = fitness_values[best_genome_idx]
#         best_program = readable_cgp_program_from_genome(
#             best_genome, meta_config["cgp_config"]
#         )

#         # print progress
#         print(f"[Meta gen {_meta_generation}] - best fitness: {best_fitness}")
#         print(best_program)

#         # NOTE - update population
#         genomes = jnp.concatenate((parents, new_genomes))

#         # NOTE - log stats
#         if wandb_run is not None:
#             wandb_run.log(
#                 {
#                     "training": {
#                         "total_emp_mean_fitness": fitness_values.mean(),
#                         "total_max_fitness": fitness_values.max(),
#                         "hc100": {
#                             "emp_mean_fit": f_hc_100.mean(),
#                             "max_fit": f_hc_100.max(),
#                         },
#                         "hc500": {
#                             "emp_mean_fit": f_hc_500.mean(),
#                             "max_fit": f_hc_500.max(),
#                         },
#                     },
#                 }
#             )
#             # Save best genome as graph and readable program
#             programm_save_path = Path(wandb_run.dir) / "programs"
#             programm_save_path.mkdir(parents=True, exist_ok=True)
#             graph_save_path = str(
#                 programm_save_path / f"gen_{_meta_generation}_best_graph.png"
#             )
#             readable_programm_save_path = str(
#                 programm_save_path / f"gen_{_meta_generation}_best_programm.txt"
#             )
#             __save_graph__(
#                 genome=best_genome,
#                 config=meta_config["cgp_config"],
#                 file=graph_save_path,
#                 input_color="green",
#                 output_color="red",
#             )
#             __write_readable_program__(
#                 genome=best_genome,
#                 config=meta_config["cgp_config"],
#                 target_file=readable_programm_save_path,
#             )
#             # Save best
#             save_path = (
#                 Path(wandb_run.dir)
#                 / "df_genomes"
#                 / f"mg_{_meta_generation}_best_genome.npy"
#             )
#             save_path = save_path.with_suffix(".npy")
#             save_path.parent.mkdir(parents=True, exist_ok=True)

#             with open(save_path, "wb") as f:
#                 jnp.save(f, best_genome)

#             wandb_run.save(
#                 str(graph_save_path), base_path=f"{wandb_run.dir}/", policy="now"
#             )
#             wandb_run.save(
#                 str(readable_programm_save_path),
#                 base_path=f"{wandb_run.dir}/",
#                 policy="now",
#             )
#             wandb_run.save(
#                   str(save_path),
#                   base_path=f"{wandb_run.dir}/", policy="now")

#         print(f"[Meta gen {_meta_generation}] - End\n")

#     return None


# def meta_learn_cgp_extended(meta_config: dict, wandb_run=None):
#     """Meta evolution of a cgp parametrized distance function"""
#     assert (
#         meta_config["cgp_config"]["n_individuals"]
#         == meta_config["evo"]["population_size"]
#     )

#     rng: jrd.KeyArray = jrd.PRNGKey(meta_config["seed"])

#     # Evaluation function based on CGP using CGP df
#     # Input size is the number of values for each neuron position vector
#     # Output size is 1, the distance between the two neurons
#     __update_config_with_data__(
#         meta_config["cgp_config"],
#         observation_space_size=meta_config["encoding"]["d"] * 2,
#         action_space_size=1,
#     )
#     # n_mutations_per_individual = int(
#     #     (meta_config["cgp_config"]["n_individuals"]
#     #     - meta_config["cgp_config"]["elite_size"])
#     #     / meta_config["cgp_config"]["elite_size"]
#     # )
#     nan_replacement = meta_config["cgp_config"]["nan_replacement"]

#     # preliminary evo steps
#     genome_mask, mutation_mask = __compute_masks__(meta_config["cgp_config"])

#     # evaluation curriculum fonctions
#     # NOTE - removed JIT
#     vec_learn_hc_100 = vmap(
#         partial(
#             learn_brax_task_cgp,
#             config=meta_config["curriculum"]["hc_100"],
#             cgp_config=meta_config["cgp_config"],
#         ),
#         in_axes=(0, None),
#     )
#     # NOTE - removed JIT
#     vec_learn_hc_500 = vmap(
#         partial(
#             learn_brax_task_cgp,
#             config=meta_config["curriculum"]["hc_500"],
#             cgp_config=meta_config["cgp_config"],
#         ),
#         in_axes=(0, None),
#     )
#     vec_learn_w2d_1000 = vmap(
#         partial(
#             learn_brax_task_cgp,
#             config=meta_config["curriculum"]["w2d_1000"],
#             cgp_config=meta_config["cgp_config"],
#         ),
#         in_axes=(0, None),
#     )

#     partial_fp_selection = partial(
#         fp_selection, n_elites=meta_config["cgp_config"]["elite_size"]
#     )
#     jit_partial_fp_selection = jit(partial_fp_selection)
#     # mutation
#     genome_transformation_function = __compute_genome_transformation_function__(
#         meta_config["cgp_config"]
#     )
#     batch_mutate_genomes = __compile_mutation__(
#         meta_config["cgp_config"],
#         genome_mask,
#         mutation_mask,
#         genome_transformation_function=genome_transformation_function,
#     )

#     # replace invalid fitness values
#     fitness_nan_replacement = jit(partial(jnp.nan_to_num, nan=nan_replacement))

#     rng, rng_generation = jrd.split(rng, 2)
#     genomes = generate_population(
#         pop_size=meta_config["cgp_config"]["n_individuals"],
#         genome_mask=genome_mask,
#         rnd_key=rng_generation,
#         genome_transformation_function=genome_transformation_function,
#     )

#     wandb_run.config.update(meta_config, allow_val_change=True)
#     for _meta_generation in range(meta_config["evo"]["n_generations"]):
#         print(f"[Meta gen {_meta_generation}] - Start")
#         rng, rng_eval_hc_100, rng_eval_hc_500, rng_w2d_1000 = jrd.split(rng, 4)
#         # NOTE - evaluate population on curriculum of tasks
#         f_hc_100 = vec_learn_hc_100(genomes, rng_eval_hc_100)
#         print(f"[Meta gen {_meta_generation}] - eval hc 100 done")
#         f_hc_500 = vec_learn_hc_500(genomes, rng_eval_hc_500)
#         print(f"[Meta gen {_meta_generation}] - eval hc 500 done")
#         f_w2d_1000 = vec_learn_w2d_1000(genomes, rng_w2d_1000)
#         print(f"[Meta gen {_meta_generation}] - eval w2d 1000 done")

#         fitness_values = f_hc_100 + f_hc_500 + f_w2d_1000

#         # NAN replacement
#         fitness_values = fitness_nan_replacement(fitness_values)

#         # NOTE - select parents
#         rng, rng_fp = jrd.split(rng, 2)
#         # Choose selection mechanism
#         parents = jit_partial_fp_selection(genomes, fitness_values, rng_fp)

#         # NOTE - compute offspring
#         rng, rng_mutation = jrd.split(rng, 2)
#         rng_multiple_mutations = jrd.split(rng_mutation, len(parents))
#         new_genomes_matrix = batch_mutate_genomes(parents, rng_multiple_mutations)
#         new_genomes = jnp.reshape(
#             new_genomes_matrix, (-1, new_genomes_matrix.shape[-1])
#         )

#         # max index
#         best_genome_idx = jnp.argmax(fitness_values)
#         best_genome = genomes[best_genome_idx]
#         best_fitness = fitness_values[best_genome_idx]
#         best_program = readable_cgp_program_from_genome(
#             best_genome, meta_config["cgp_config"]
#         )

#         # print progress
#         print(f"[Meta gen {_meta_generation}] - best fitness: {best_fitness}")
#         print(best_program)

#         # NOTE - update population
#         genomes = jnp.concatenate((parents, new_genomes))

#         # NOTE - log stats
#         if wandb_run is not None:
#             wandb_run.log(
#                 {
#                     "training": {
#                         "total_emp_mean_fitness": fitness_values.mean(),
#                         "total_max_fitness": fitness_values.max(),
#                         "hc100": {
#                             "emp_mean_fit": f_hc_100.mean(),
#                             "max_fit": f_hc_100.max(),
#                         },
#                         "hc500": {
#                             "emp_mean_fit": f_hc_500.mean(),
#                             "max_fit": f_hc_500.max(),
#                         },
#                         "w2d1000": {
#                             "emp_mean_fit": f_w2d_1000.mean(),
#                             "max_fit": f_w2d_1000.max(),
#                         },
#                     },
#                 }
#             )
#             # Save best genome as graph and readable program
#             programm_save_path = Path(wandb_run.dir) / "programs"
#             programm_save_path.mkdir(parents=True, exist_ok=True)
#             graph_save_path = str(
#                 programm_save_path / f"gen_{_meta_generation}_best_graph.png"
#             )
#             readable_programm_save_path = str(
#                 programm_save_path / f"gen_{_meta_generation}_best_programm.txt"
#             )
#             __save_graph__(
#                 genome=best_genome,
#                 config=meta_config["cgp_config"],
#                 file=graph_save_path,
#                 input_color="green",
#                 output_color="red",
#             )
#             __write_readable_program__(
#                 genome=best_genome,
#                 config=meta_config["cgp_config"],
#                 target_file=readable_programm_save_path,
#             )
#             # Save best
#             save_path = (
#                 Path(wandb_run.dir)
#                 / "df_genomes"
#                 / f"mg_{_meta_generation}_best_genome.npy"
#             )
#             save_path = save_path.with_suffix(".npy")
#             save_path.parent.mkdir(parents=True, exist_ok=True)

#             with open(save_path, "wb") as f:
#                 jnp.save(f, best_genome)

#             wandb_run.save(
#                 str(graph_save_path), base_path=f"{wandb_run.dir}/", policy="now"
#             )
#             wandb_run.save(
#                 str(readable_programm_save_path),
#                 base_path=f"{wandb_run.dir}/",
#                 policy="now",
#             )
#             wandb_run.save(str(save_path),
#               base_path=f"{wandb_run.dir}/", policy="now")

#         print(f"[Meta gen {_meta_generation}] - End\n")

#     return None


# def meta_learn_cgp_evosax(meta_config: dict, wandb_run=None):
#     """Gymnax-only environnments meta-learning (cartpole, Acrobot) for a
#     cgp parametrized distance function
#     """
#     rng = jrd.PRNGKey(meta_config["seed"])

#     # Evaluation function based on CGP using CGP df
#     # Input size is the number of values for each neuron position vector
#     # Output size is 1, the distance between the two neurons
#     __update_config_with_data__(
#         meta_config["cgp_config"],
#         observation_space_size=meta_config["encoding"]["d"] * 2,
#         action_space_size=1,
#     )
#     n_mutations_per_individual = int(
#         (
#             meta_config["cgp_config"]["n_individuals"]
#             - meta_config["cgp_config"]["elite_size"]
#         )
#         / meta_config["cgp_config"]["elite_size"]
#     )
#     nan_replacement = meta_config["cgp_config"]["nan_replacement"]

#     # preliminary evo steps
#     genome_mask = compute_cgp_genome_mask(
#         meta_config["cgp_config"],
#         n_in=meta_config["cgp_config"]["n_in"],
#         n_out=meta_config["cgp_config"]["n_out"],
#     )
#     mutation_mask = compute_cgp_mutation_prob_mask(
#         meta_config["cgp_config"], n_out=meta_config["cgp_config"]["n_out"]
#     )

#     # evaluation
#     vec_learn_cartpole = jit(
#         vmap(
#             partial(
#                 learn_gymnax_task_cgp_df_mean,
#                 config=meta_config["curriculum"]["cart"],
#                 cgp_config=meta_config["cgp_config"],
#             ),
#             in_axes=(0, None),
#         )
#     )
#     vec_learn_acrobot = jit(
#         vmap(
#             partial(
#                 learn_gymnax_task_cgp_df_mean,
#                 config=meta_config["curriculum"]["acrobot"],
#                 cgp_config=meta_config["cgp_config"],
#             ),
#             in_axes=(0, None),
#         )
#     )

#     # fp select
#     partial_fp_selection = partial(
#         fp_selection, n_elites=meta_config["cgp_config"]["elite_size"]
#     )
#     jit_partial_fp_selection = jit(partial_fp_selection)
#     # mutation
#     partial_multiple_mutations = partial(
#         mutate_genome_n_times,
#         n_mutations=n_mutations_per_individual,
#         genome_mask=genome_mask,
#         mutation_mask=mutation_mask,
#     )
#     vmap_multiple_mutations = vmap(partial_multiple_mutations)
#     jit_vmap_multiple_mutations = jit(vmap_multiple_mutations)
#     # replace invalid fitness values
#     fitness_replacement = jit(partial(jnp.nan_to_num, nan=nan_replacement))

#     rng, rng_generation = jrd.split(rng, 2)
#     genomes = generate_population(
#         pop_size=meta_config["cgp_config"]["n_individuals"],
#         genome_mask=genome_mask,
#         rnd_key=rng_generation,
#     )

#     for _meta_generation in range(meta_config["evo"]["n_generations"]):
#         print(f"[Meta gen {_meta_generation}] - Start")
#         rng, rng_cartpole, rng_acrobot = jrd.split(rng, 3)
#         # NOTE - evaluate population on curriculum
#         f_cartpole = vec_learn_cartpole(genomes, rng_cartpole)
#         max_f_cartpole = jnp.max(f_cartpole)

#         f_acrobot = (
#             vec_learn_acrobot(genomes, rng_acrobot) if max_f_cartpole > 400 else 0
#         )
#         max_f_acrobot = jnp.max(f_acrobot)

#         fitness_values = f_cartpole + f_acrobot
#         # fix nan values
#         fitness_values = fitness_replacement(fitness_values)

#         # NOTE - select parents
#         rng, rng_fp = jrd.split(rng, 2)
#         # Choose selection mechanism
#         parents = jit_partial_fp_selection(genomes, fitness_values, rng_fp)

#         # NOTE - compute offspring
#         rng, rng_mutation = jrd.split(rng, 2)
#         rng_multiple_mutations = jrd.split(rng_mutation, len(parents))
#         new_genomes_matrix = jit_vmap_multiple_mutations(
#             parents, rng_multiple_mutations
#         )
#         new_genomes = jnp.reshape(
#             new_genomes_matrix, (-1, new_genomes_matrix.shape[-1])
#         )

#         # max index
#         best_genome_idx = jnp.argmax(fitness_values)
#         best_genome = genomes[best_genome_idx]
#         best_fitness = fitness_values[best_genome_idx]
#         best_program = readable_cgp_program_from_genome(
#             best_genome, meta_config["cgp_config"]
#         )

#         # print progress
#         print(f"[Meta gen {_meta_generation}] - best fitness: {best_fitness}")
#         print(f"\t{max_f_cartpole}")
#         print(f"\t{max_f_acrobot}")
#         print(best_program)

#         if wandb_run is not None:
#             wandb_run.log({"best_fitness": best_fitness})

#         # NOTE - update population
#         genomes = jnp.concatenate((parents, new_genomes))
#         print(f"[Meta gen {_meta_generation}] - End\n")

#     return None


def create_l2_indiv(meta_config: dict):
    # NOTE - replace some indiv with l2 indiv
    # - padd zeroes until n_nodes reached for each of those
    # - concat [x, y, f, out]
    # - Can append to the population (check, better replace&)
    x_genes = jnp.array([0, 1, 2, 9, 10, 11, 12, 15, 16] + [17, 18, 19, 20, 21])
    y_genes = jnp.array([3, 4, 5, 9, 10, 11, 13, 14, 0] + [6, 6, 6, 6, 6])
    f_genes = jnp.array([1, 1, 1, 2, 2, 2, 0, 0, 11] + [1, 1, 1, 1, 1])
    out = jnp.array([22])

    padded_x_genes = jnp.pad(
        x_genes,
        (0, meta_config["cgp_config"]["n_nodes"] - x_genes.shape[0]),
        mode="constant",
        constant_values=0,
    )
    padded_y_genes = jnp.pad(
        y_genes,
        (0, meta_config["cgp_config"]["n_nodes"] - y_genes.shape[0]),
        mode="constant",
        constant_values=0,
    )
    padded_f_genes = jnp.pad(
        f_genes,
        (0, meta_config["cgp_config"]["n_nodes"] - f_genes.shape[0]),
        mode="constant",
        constant_values=0,
    )
    new_base_indiv = jnp.concatenate(
        (padded_x_genes, padded_y_genes, padded_f_genes, out)
    )
    return new_base_indiv


def evaluate_used_inputs(genome, rng, cgp_config: dict, d: int = 6):
    """Evaluate the genome, checking if the input nodes are all being used.
    Different input values are generated and the output is analyzed."""
    program = genome_to_cgp_program(
        genome=genome,
        config=cgp_config,
        # output wrapper by default is tanh
        outputs_wrapper=lambda e: e,
    )

    initial_input = jrd.normal(rng, (d,))

    _, initial_output = program(initial_input, jnp.zeros(cgp_config["buffer_size"]))

    total_dts = []
    for in_pos in range(d):
        # for pert_value in [0.5, -0.5, 1., -1., 2., -2.]:
        for pert_value in [1.0, -1.0]:
            # NOTE - vérifier qu'en changeant la valeur d'entrée, la sortie change aussi
            perturbed_input = initial_input.at[in_pos].add(pert_value)
            _, in_pos_perturbation_out = program(
                perturbed_input, jnp.zeros(cgp_config["buffer_size"])
            )
            dt = jnp.abs(initial_output - in_pos_perturbation_out)
            total_dts.append(dt)

    # 1 if all dt's are different from 0
    # 0 if at least one dt is 0
    non_z = jnp.count_nonzero(jnp.array(total_dts))
    fit_term = jnp.clip(
        non_z - (len(total_dts) - 1),
        a_min=0,
        a_max=1,
    )
    return fit_term


def meta_learn_cgp_corrected(meta_config: dict, wandb_run=None, beta: float = 0.5):
    """Meta evolution of a cgp parametrized distance function,
    using a corrected fitness function forcing the policy neural networks to
    enforce some basic properties."""
    assert (
        meta_config["cgp_config"]["n_individuals"]
        == meta_config["evo"]["population_size"]
    )

    rng: jrd.KeyArray = jrd.PRNGKey(meta_config["seed"])
    rng, rng_gen_pop, rng_gen_idx = jrd.split(rng, 3)

    # NOTE - CGP preliminary steps
    # Evaluation function based on CGP using CGP df
    # Input size is the number of values for each neuron position vector
    # Output size is 1, the distance between the two neurons
    __update_config_with_data__(
        meta_config["cgp_config"],
        observation_space_size=meta_config["encoding"]["d"] * 2,
        action_space_size=1,
    )
    genome_mask, mutation_mask = __compute_masks__(meta_config["cgp_config"])
    parents_selection = __compile_parents_selection__(meta_config["cgp_config"])
    select_survivals = __compile_survival_selection__(meta_config["cgp_config"])
    _genome_transformation_function = __compute_genome_transformation_function__(
        meta_config["cgp_config"]
    )
    batch_mutate_genomes = __compile_mutation__(
        meta_config["cgp_config"],
        genome_mask,
        mutation_mask,
        genome_transformation_function=_genome_transformation_function,
    )
    nan_replacement = meta_config["cgp_config"]["nan_replacement"]
    fitness_nan_replacement = jit(partial(jnp.nan_to_num, nan=nan_replacement))
    # Generate the base population
    genomes = generate_population(
        pop_size=meta_config["cgp_config"]["n_individuals"],
        genome_mask=genome_mask,
        rnd_key=rng_gen_pop,
        genome_transformation_function=_genome_transformation_function,
    )
    _l2_indiv = create_l2_indiv(meta_config)
    assert _l2_indiv.shape[0] == genomes.shape[-1]
    # Randomly changes individuals of the base population to the handcrafted individual
    for idx in jrd.randint(
        rng_gen_idx,
        shape=(int(meta_config["cgp_config"]["n_individuals"] / 8),),
        minval=0,
        maxval=meta_config["cgp_config"]["n_individuals"],
    ):
        genomes = genomes.at[idx].set(_l2_indiv)

    # NOTE - Evaluation steps: NN prop enforce & Policy evaluation
    vec_evaluate_network_properties = jit(
        vmap(
            partial(
                evaluate_rand_network_properties_n_times,
                meta_config=meta_config,
                df_type="cgp",
                n=32,
            ),
            in_axes=(0, 0),
        )
    )
    # NOTE - Evaluate the nomber of input nodes used by cgp
    vec_evaluate_used_inputs = jit(
        vmap(
            partial(evaluate_used_inputs, cgp_config=meta_config["cgp_config"], d=6),
            in_axes=(0, None),
        ),
    )
    if beta < 1:
        # evaluation curriculum fonctions
        vec_learn_hc_500 = vmap(
            partial(
                learn_brax_task_cgp,
                config=meta_config["curriculum"]["hc_500"],
                cgp_config=meta_config["cgp_config"],
            ),
            in_axes=(0, None),
        )
        vec_learn_brax_task_cgp_d0_50 = vmap(
            partial(
                learn_brax_task_cgp_d0_50,
                config=meta_config["curriculum"]["hc_500"],
                cgp_config=meta_config["cgp_config"],
            ),
            in_axes=(0, None),
        )

    if wandb_run is not None:
        wandb_run.config.update(meta_config, allow_val_change=True)

    for _meta_generation in range(meta_config["evo"]["n_generations"]):
        print(f"[Meta gen {_meta_generation}] - Start")

        (
            rng,
            rng_eval_hc500,
            rng_eval_net_prop,
            rng_used_inputs,
            rng_survival,
        ) = jrd.split(rng, 5)
        rng_eval_net_prop = jrd.split(
            key=rng_eval_net_prop, num=meta_config["evo"]["population_size"]
        )

        # SECTION - evaluate population on nn properties and curriculum of tasks
        f_expr, f_w_distr, f_inp = vec_evaluate_network_properties(
            genomes, rng_eval_net_prop
        )
        f_expr = fitness_nan_replacement(f_expr)
        f_w_distr = fitness_nan_replacement(f_w_distr)
        f_inp = fitness_nan_replacement(f_inp)
        f_net_prop = (
            min_max_scaler(f_expr) + min_max_scaler(f_w_distr) + min_max_scaler(f_inp)
        )

        if beta < 1:
            # NOTE - Policy evaluation
            f_hc_500_max = vec_learn_hc_500(genomes, rng_eval_hc500)
            f_hc_500_d0_50 = vec_learn_brax_task_cgp_d0_50(genomes, rng_eval_hc500)

            f_policy_eval = min_max_scaler(f_hc_500_max) + min_max_scaler(
                f_hc_500_d0_50
            )
        else:
            f_policy_eval = 0.0

        # NOTE - add penalty for small programms
        active_node_sizes = jnp.array(
            [
                compute_active_size(genome, meta_config["cgp_config"])[0]
                for genome in genomes
            ]
        )
        n_total_nodes = genomes.shape[-1]
        fit_active_node_sizes = jnp.exp(-(active_node_sizes / n_total_nodes) / 0.1)
        # NOTE - add fitness term for nomber of input nodes used
        fit_used_input_nodes, _ = vec_evaluate_used_inputs(genomes, rng_used_inputs)

        fitness_cgp_extra = -fit_active_node_sizes + fit_used_input_nodes

        fitness_values = (
            beta * f_net_prop + (1 - beta) * f_policy_eval + fitness_cgp_extra
        )
        assert fitness_values is not None
        # !SECTION

        # NAN replacement
        fitness_values = fitness_nan_replacement(fitness_values)

        # NOTE - select parents
        rng, rng_p_sel = jrd.split(rng, 2)
        parents = parents_selection(genomes, fitness_values, rng_p_sel)

        # NOTE - compute offspring
        rng, rng_mutation = jrd.split(rng, 2)
        rng_multiple_mutations = jrd.split(rng_mutation, len(parents))
        new_genomes_matrix = batch_mutate_genomes(parents, rng_multiple_mutations)
        new_genomes = jnp.reshape(
            new_genomes_matrix, (-1, new_genomes_matrix.shape[-1])
        )

        # get best individual
        best_genome_idx = jnp.argmax(fitness_values)
        best_genome = genomes[best_genome_idx]
        best_fitness = fitness_values[best_genome_idx]
        best_program = readable_cgp_program_from_genome(
            best_genome, meta_config["cgp_config"]
        )

        # NOTE - update population
        survivals = select_survivals(genomes, fitness_values, rng_survival)
        genomes = jnp.concatenate((survivals, new_genomes))

        # print progress
        print(f"[Meta gen {_meta_generation}] - best fitness: {best_fitness}")
        print(best_program)

        # NOTE - log stats
        if wandb_run is not None:
            to_log = {
                "training": {
                    "total_emp_mean_fitness": fitness_values.mean(),
                    "total_max_fitness": fitness_values.max(),
                    "net_prop": {
                        "f_expressivity": f_expr.mean(),
                        "f_weight_distribution": f_w_distr.mean(),
                        "f_input_restoration": f_inp.mean(),
                    },
                    "cgp_extra": {
                        "all_nodes_used": fit_used_input_nodes.mean(),
                        "best_all_nodes_used": fit_used_input_nodes[best_genome_idx],
                        "fit_active_node_sizes": fit_active_node_sizes.mean(),
                        "active_node_sizes": active_node_sizes.mean(),
                    },
                },
            }
            if beta < 1:
                to_log["training"]["hc500"] = {
                    "emp_mean_fit": f_hc_500_max.mean(),
                    "max_fit": f_hc_500_max.max(),
                }
                to_log["training"]["hc500_d0_50"] = {
                    "emp_mean_fit": f_hc_500_d0_50.mean(),
                    "max": f_hc_500_d0_50.max(),
                }
            wandb_run.log(to_log)
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
                config=meta_config["cgp_config"],
                file=graph_save_path,
                input_color="green",
                output_color="red",
            )
            __write_readable_program__(
                genome=best_genome,
                config=meta_config["cgp_config"],
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
