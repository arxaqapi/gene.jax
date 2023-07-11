from pathlib import Path

import jax.numpy as jnp
import wandb

from gene.tracker import Tracker
from gene.learning import learn_brax_task
from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.visualize.la import run_fla_brax
from gene.core.distances import get_df
from gene.core.models import get_model
from gene.core.decoding import get_decoder


class Experiment:
    """Perform an experiment composed of a set of runs,
    the corresponding FLA, and the visualization of the result.
    """

    def __init__(self, config: dict, project_name) -> None:
        self.config = config
        self.project_name = project_name

    def run(self, seed: int):
        self.config["seed"] = seed

        wdb_run = wandb.init(
            project=self.project_name, config=self.config, tags=["single"]
        )

        df = get_df(self.config)()
        tracker, tracker_state = learn_brax_task(
            self.config,
            df=df,
            wdb_run=wdb_run,
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
            decoder=get_decoder(self.config)(self.config, df),
            wdb_run=wdb_run,
        )

        # NOTE - visualize the learned brax genome (best and sample_mean)
        viz_save_path = Path(wdb_run.dir) / "viz"
        viz_save_path.mkdir(parents=True, exist_ok=True)
        # best overall individual (get_top_k_genomes)
        render_brax(
            viz_save_path / "learned_best_individual",
            *visualize_brax(
                config=self.config,
                genome=tracker.get_top_k_genomes(tracker_state)[0],
                model=get_model(self.config),
                df=df,
            ),
        )
        print("[Check] - Viz best ok")
        # last mean individual
        render_brax(
            viz_save_path / "learned_last_mean_individual",
            *visualize_brax(
                config=self.config,
                genome=tracker.get_final_center_individual(tracker_state),
                model=get_model(self.config),
                df=df,
            ),
        )
        print("[Check] - Viz mean ok")

        wdb_run.finish()
        print("[Check] - run finished")

        return mean_fitness, best_fitness

    def run_n(self, seeds: list[int]) -> list[Tracker]:
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
