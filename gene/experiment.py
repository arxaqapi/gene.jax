from pathlib import Path

import jax.numpy as jnp
import jax.random as jrd
import wandb

from gene.tracker import Tracker
from gene.learning import learn_brax_task
from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.core.distances import Distance_functions
from gene.core.models import get_model


class Experiment:
    """Perform an experiment composed of a set of runs,
    the corresponding LLA, and the visualization of the result.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self):
        raise NotImplementedError

    # FIXME - move subset runs to run()
    # FIXME - fix errors in LLA (implement)
    # FIXME - fix errors in Visualization
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
            # NOTE - Perform LLA of the run
            raise NotImplementedError

            # NOTE - visualize the learned brax genome (best and sample_mean)
            path = Path("html")
            path.mkdir(exist_ok=True)

            render_brax(
                *visualize_brax(
                    self.config,
                    genome,
                    model=get_model(self.config),
                    df=Distance_functions[self.config["encoding"]["distance"]](),
                    rng=jrd.PRNGKey(None),
                ),
                path / f"run_{run_id.split('/')[-1]}_{genome_id[:-4]}",
            )

            wdb_run.finish()

        stats = {
            "mean_mean_fitnesses": jnp.array(mean_fitnesses).mean(),
            "var_mean_fitnesses": jnp.array(mean_fitnesses).var(),
            "mean_best_fitnesses": jnp.array(best_fitnesses).mean(),
            "var_best_fitnesses": jnp.array(best_fitnesses).var(),
        }

        return stats
