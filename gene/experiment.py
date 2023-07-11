from pathlib import Path

import jax.numpy as jnp
import jax.random as jrd
import wandb

from gene.tracker import Tracker
from gene.learning import learn_brax_task
from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.visualize.la import run_fla_brax
from gene.core.distances import Distance_functions
from gene.core.models import get_model


class Experiment:
    """Perform an experiment composed of a set of runs,
    the corresponding LLA, and the visualization of the result.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, seed: int):
        self.config["seed"] = seed

        wdb_run = wandb.init(
            project="Brax expe bench test", config=self.config, tags=["single"]
        )

        tracker, tracker_state = learn_brax_task(
            self.config,
            df=Distance_functions[self.config["encoding"]["distance"]](),
            wdb_run=wdb_run,
        )
        print("[Check] - Learning loop ok")

        mean_fitness = tracker_state["eval"]["mean_fit"]
        # Get best individual [0] from last generation [-1]
        best_fitness = tracker_state["training"]["top_k_fit"][-1][0]

        # NOTE - Perform LLA of the run, get genomes from state
        run_fla_brax(
            config=self.config,
            plot_title=f"{self.config['encoding']['type']} encoding \
                            w. {self.config['encoding']['distance']} distance \
                            on {self.config['task']['environnment']}",
            initial_genome=tracker.get_initial_center_individual(tracker_state),
            final_genome=tracker.get_final_center_individual(tracker_state),
            wandb_run=wdb_run,
        )
        print("[Check] - LLA OK")

        # NOTE - visualize the learned brax genome (best and sample_mean)
        path = Path("html")
        path.mkdir(exist_ok=True)
        # best overall individual (get_top_k_genomes)
        render_brax(
            *visualize_brax(
                config=self.config,
                genome=tracker.get_top_k_genomes[0],
                model=get_model(self.config),
                df=Distance_functions[self.config["encoding"]["distance"]](),
                rng=jrd.PRNGKey(None),
            ),
            path / "learned_best_individual",
        )
        print("[Check] - Viz best ok")
        # last mean individual
        render_brax(
            *visualize_brax(
                config=self.config,
                genome=tracker.get_final_genome[0],
                model=get_model(self.config),
                df=Distance_functions[self.config["encoding"]["distance"]](),
                rng=jrd.PRNGKey(None),
            ),
            path / "learned_last_mean_individual",
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
