from pathlib import Path
from typing import Union
from copy import deepcopy
import time

import wandb
import jax.numpy as jnp

from gene.learning import learn_brax_task
from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.visualize.la import run_fla_brax
from gene.visualize.neurons import visualize_neurons_3d, visualize_neurons_2d
from gene.core.distances import get_df, DistanceFunction, NNDistance, CGPDistance
from gene.core.models import get_model
from gene.core.decoding import get_decoder
from gene.utils import validate_json, make_wdb_subfolder, meta_save_genome

from cgpax.analysis.genome_analysis import __save_graph__


class Experiment:
    """Perform an experiment composed of a set of runs,
    the corresponding FLA, and the visualization of the result.
    """

    def __init__(
        self,
        config: dict,
        wandb_run,
        distance_function: Union[DistanceFunction, None] = None,
    ) -> None:
        self.config = config
        self.wandb_run = wandb_run
        self.df = (
            get_df(self.config)() if distance_function is None else distance_function
        )

    def run(
        self,
        seed: Union[int, None] = None,
        save_step: int = 2000,
        run_fla: bool = False,
    ):
        if seed is not None:
            self.config["seed"] = seed

        tracker, tracker_state = learn_brax_task(
            self.config,
            df=self.df,
            wdb_run=self.wandb_run,
            save_step=save_step,
            es_param_dict=self.config["evo"].get("es_param_dict", {}),
        )

        mean_fitness = tracker_state["eval"]["mean_fit"]
        # Get best individual [0] from last generation [-1]
        best_fitness = tracker_state["training"]["top_k_fit"][-1][0]

        if run_fla:
            # NOTE - Perform FLA of the run, get genomes from state
            run_fla_brax(
                plot_title=f"{self.config['encoding']['type']} encoding \
                                w. {self.config['encoding']['distance']} distance \
                                on {self.config['task']['environnment']}",
                config=self.config,
                initial_genome=tracker.get_initial_center_individual(tracker_state),
                final_genome=tracker.get_final_center_individual(tracker_state),
                decoder=get_decoder(self.config)(self.config, self.df),
                wdb_run=self.wandb_run,
            )

        # NOTE - visualize the learned brax genome (best and sample_mean)
        viz_save_path = Path(self.wandb_run.dir) / "viz"
        viz_save_path.mkdir(parents=True, exist_ok=True)
        # best overall individual (get_top_k_genomes)
        render_brax(
            viz_save_path / "learned_best_individual",
            *visualize_brax(
                config=self.config,
                genome=tracker.get_top_k_genomes(tracker_state)[0],
                model=get_model(self.config),
                df=self.df,
            ),
        )
        # last mean individual
        render_brax(
            viz_save_path / "learned_last_mean_individual",
            *visualize_brax(
                config=self.config,
                genome=tracker.get_final_center_individual(tracker_state),
                model=get_model(self.config),
                df=self.df,
            ),
        )

        # NOTE - visualize neurons positions in 3D space
        # get appropriate function
        if (
            self.config["encoding"]["d"] in [2, 3]
            and self.config["encoding"]["type"] == "gene"
        ):
            visualize_neurons = (
                visualize_neurons_2d
                if self.config["encoding"]["d"] == 2
                else visualize_neurons_3d
            )

            neuron_pos_path = Path(self.wandb_run.dir) / "neurons_positions"
            neuron_pos_path.mkdir(parents=True, exist_ok=True)
            # start mean
            visualize_neurons(
                tracker.get_initial_center_individual(tracker_state),
                self.config,
                title=neuron_pos_path / "initial_neuron_positions",
            )
            # final mean
            visualize_neurons(
                tracker.get_final_center_individual(tracker_state),
                self.config,
                title=neuron_pos_path / "final_neuron_positions",
            )
            # best individuals
            for k in range(tracker.top_k):
                visualize_neurons(
                    tracker.get_top_k_genomes(tracker_state)[k],
                    self.config,
                    title=neuron_pos_path / f"top_{k}_neuron_positions",
                )

        self.wandb_run.finish()

        return mean_fitness, best_fitness

    def run_n(self, seeds: list[int]):
        """Run n experiments, conditioned by the number of seeds provided,
        and returns all statistics.

        Args:
            seed (list[int]): List of seeds used to run the experiments
        """

        mean_fitnesses = []
        best_fitnesses = []

        for seed in seeds:
            mean_fit, best_fit = self.run(seed)
            mean_fitnesses.append(mean_fit)
            best_fitnesses.append(best_fit)

        stats = {
            "mean_mean_fitnesses": jnp.array(mean_fitnesses).mean(),
            "var_mean_fitnesses": jnp.array(mean_fitnesses).var(),
            "mean_best_fitnesses": jnp.array(best_fitnesses).mean(),
            "var_best_fitnesses": jnp.array(best_fitnesses).var(),
        }

        return stats


def comparison_experiment(
    config: dict,
    nn_df_genome,
    nn_df_model_config: dict,
    seeds: list[int] = [56789, 98712, 1230],
    project: str = "devnull",
    expe_time=None,
    extra_tags: list[str] = [],
):
    """Task agnostic run expe"""
    if expe_time is None:
        expe_time = int(time.time())

    for seed in seeds:
        # NOTE - config setup
        base_config = deepcopy(config)
        base_config["task"]["episode_length"] = 1000
        base_config["evo"]["population_size"] = 256
        base_config["seed"] = seed

        # NOTE - 2. Use learned distance function to train a policy
        nn_df_config = deepcopy(base_config)
        nn_df_config["encoding"]["distance"] = ""
        nn_df_config["group"] = "learned"
        validate_json(nn_df_config)

        with wandb.init(
            project=project,
            name="CC-Comp-learned-nn",
            config=nn_df_config,
            tags=[f"{expe_time}"] + extra_tags,
        ) as wdb_nn_df:
            Experiment(
                nn_df_config,
                wdb_nn_df,
                distance_function=NNDistance(
                    distance_genome=nn_df_genome,
                    config=nn_df_model_config,
                    nn_layers_dims=nn_df_model_config["net"]["layer_dimensions"],
                ),
            ).run()

        # NOTE - 3.1. GENE w. pL2
        conf_gene_pl2 = deepcopy(base_config)
        conf_gene_pl2["encoding"]["distance"] = "pL2"
        conf_gene_pl2["encoding"]["type"] = "gene"
        conf_gene_pl2["group"] = "pL2"
        validate_json(conf_gene_pl2)

        with wandb.init(
            project=project,
            name="CC-Comp-pL2",
            config=conf_gene_pl2,
            tags=[f"{expe_time}"] + extra_tags,
        ) as wdb_gene_pl2:
            Experiment(
                conf_gene_pl2,
                wdb_gene_pl2,
            ).run()

        # NOTE - 3.1. GENE w. L2
        conf_gene_l2 = deepcopy(base_config)
        conf_gene_l2["encoding"]["distance"] = "L2"
        conf_gene_l2["encoding"]["type"] = "gene"
        conf_gene_l2["group"] = "L2"
        validate_json(conf_gene_l2)

        with wandb.init(
            project=project,
            name="CC-Comp-L2",
            config=conf_gene_l2,
            tags=[f"{expe_time}"] + extra_tags,
        ) as wdb_gene_l2:
            Experiment(
                conf_gene_l2,
                wdb_gene_l2,
            ).run()

        # NOTE - 3.1. Direct
        conf_direct = deepcopy(base_config)
        conf_direct["encoding"]["type"] = "direct"
        conf_direct["encoding"]["distance"] = "pL2"
        conf_direct["group"] = "direct"
        validate_json(conf_direct)

        with wandb.init(
            project=project,
            name="CC-Comp-direct",
            config=conf_direct,
            tags=[f"{expe_time}"] + extra_tags,
        ) as wdb_direct:
            Experiment(
                conf_direct,
                wdb_direct,
            ).run()


def comparison_experiment_cgp(
    config: dict,
    cgp_config: dict,
    cgp_df_genome_archive: dict,
    project: str,
    entity: str,
    seeds: list[int] = [56789, 98712, 1230],
    expe_time=None,
    extra_tags: list[str] = [],
):
    """Takes a cgp distance function genome, evaluates it and compares it to
    gene encoding with pL2 and L2, direct encoding

    Args:
        config (dict): config dict, containing the task to evaluate on
        cgp_config (dict): config dict of the CGP distance function
        cgp_df_genome_archive (dict): genome of the learned CGP distance funtion
            to be evaluated
        project (str): Project name for weights and biases
        seeds (list[int], optional): _description_. Defaults to [56789, 98712, 1230].
        expe_time (_type_, optional): Time tag for grouping runs. Defaults to None.
        extra_tags (list[str], optional): tags to be added to the runs. Defaults to [].
    """
    if expe_time is None:
        expe_time = int(time.time())

    for seed in seeds:
        # NOTE - config setup
        base_config = deepcopy(config)
        base_config["task"]["episode_length"] = 1000
        base_config["evo"]["population_size"] = 256
        base_config["seed"] = seed

        # NOTE - 2. Use learned distance function to train a policy
        cgp_df_config = deepcopy(base_config)
        cgp_df_config["encoding"]["distance"] = ""
        cgp_df_config["group"] = "learned"
        validate_json(cgp_df_config)

        # NOTE - Perform 4 runs with different best programs
        for i, gen_archive in enumerate(
            list(reversed(cgp_df_genome_archive.values()))[:2]
        ):
            for j, archived_genome in enumerate(gen_archive["top_3"][:2]):
                with wandb.init(
                    project=project,
                    entity=entity,
                    name=f"CC-cgp-learned-{i}-{j}",
                    config=cgp_df_config,
                    tags=[f"{expe_time}"] + extra_tags,
                ) as wdb_cgp_df:
                    Experiment(
                        cgp_df_config,
                        wdb_cgp_df,
                        distance_function=CGPDistance(
                            cgp_genome=archived_genome,
                            cgp_config=cgp_config,
                        ),
                    ).run()

                    # NOTE - Save cgp graph
                    program_save_path = make_wdb_subfolder(wdb_cgp_df, "cgp_df")
                    graph_save_path = str(
                        program_save_path / f"graph_of_cgp_df_id_{config['epoch_id']}.png"
                    )
                    __save_graph__(
                        genome=archived_genome,
                        config=cgp_config,
                        file=graph_save_path,
                        input_color="green",
                        output_color="red",
                    )
                    meta_save_genome(graph_save_path, wdb_cgp_df)

        # NOTE - 3.1. GENE w. pL2
        conf_gene_pl2 = deepcopy(base_config)
        conf_gene_pl2["encoding"]["distance"] = "pL2"
        conf_gene_pl2["encoding"]["type"] = "gene"
        conf_gene_pl2["group"] = "pL2"
        validate_json(conf_gene_pl2)

        with wandb.init(
            project=project,
            entity=entity,
            name="CC-pL2",
            config=conf_gene_pl2,
            tags=[f"{expe_time}"] + extra_tags,
        ) as wdb_gene_pl2:
            Experiment(
                conf_gene_pl2,
                wdb_gene_pl2,
            ).run()

        # NOTE - 3.1. GENE w. L2
        conf_gene_l2 = deepcopy(base_config)
        conf_gene_l2["encoding"]["distance"] = "L2"
        conf_gene_l2["encoding"]["type"] = "gene"
        conf_gene_l2["group"] = "L2"
        validate_json(conf_gene_l2)

        with wandb.init(
            project=project,
            entity=entity,
            name="CC-L2",
            config=conf_gene_l2,
            tags=[f"{expe_time}"] + extra_tags,
        ) as wdb_gene_l2:
            Experiment(
                conf_gene_l2,
                wdb_gene_l2,
            ).run()

        # NOTE - 3.1. Direct
        conf_direct = deepcopy(base_config)
        conf_direct["encoding"]["type"] = "direct"
        conf_direct["encoding"]["distance"] = "pL2"
        conf_direct["group"] = "direct"
        validate_json(conf_direct)

        with wandb.init(
            project=project,
            entity=entity,
            name="CC-direct",
            config=conf_direct,
            tags=[f"{expe_time}"] + extra_tags,
        ) as wdb_direct:
            Experiment(
                conf_direct,
                wdb_direct,
            ).run()
