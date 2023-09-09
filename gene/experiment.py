from pathlib import Path
from typing import Union

import jax.numpy as jnp

from gene.learning import learn_brax_task
from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.visualize.la import run_fla_brax
from gene.visualize.neurons import visualize_neurons_3d, visualize_neurons_2d
from gene.core.distances import get_df, DistanceFunction
from gene.core.models import get_model
from gene.core.decoding import get_decoder


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

    def run(self, seed: Union[int, None] = None, save_step: int = 2000):
        if seed is not None:
            self.config["seed"] = seed

        tracker, tracker_state = learn_brax_task(
            self.config, df=self.df, wdb_run=self.wandb_run, save_step=save_step
        )

        mean_fitness = tracker_state["eval"]["mean_fit"]
        # Get best individual [0] from last generation [-1]
        best_fitness = tracker_state["training"]["top_k_fit"][-1][0]

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
