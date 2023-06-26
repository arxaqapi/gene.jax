import jax.numpy as jnp
import wandb

from gene.tracker import Tracker
from gene.learning import learn_brax_task
from gene.core.distances import Distance_functions


class Experiment:
    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self):
        raise NotImplementedError

    def run_n(self, seeds: list[int]) -> list[Tracker]:
        """Run n experiments, conditioned by the number of seeds provided,
        and returns all statistics.

        Args:
            seed (list[int]): List of seeds used to run the experiments
        """

        mean_fitnesses = []
        best_fitnesses = []

        for seed in seeds:
            self.config["seed"] = seed

            wdb_run = wandb.init(
                project="Brax expe bench test", config=self.config, tags=["single"]
            )

            tracker_state = learn_brax_task(
                self.config,
                df=Distance_functions[self.config["encoding"]["distance"]](),
                wdb_run=wdb_run,
            )

            mean_fitnesses.append(tracker_state["eval"]["mean_fit"])
            # Get best individual [0] from last generation [-1]
            best_fitnesses.append(tracker_state["training"]["top_k_fit"][-1][0])

            wdb_run.finish()

        stats = {
            "mean_mean_fitnesses": jnp.array(mean_fitnesses).mean(),
            "var_mean_fitnesses": jnp.array(mean_fitnesses).var(),
            "mean_best_fitnesses": jnp.array(best_fitnesses).mean(),
            "var_best_fitnesses": jnp.array(best_fitnesses).var(),
        }

        return stats
