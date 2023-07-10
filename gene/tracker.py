from functools import partial
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import jax.random as jrd
from jax import jit, tree_util
import chex

from gene.core.decoding import Decoder

TrackerState = chex.ArrayTree


class Tracker:
    """A minimal tracker object that keeps track of some metrics and
    key individuals during a run.

    ### Metrics tracked
    - top k fitnesses encountered (*)
    - fitness of the sample mean of the es
    ### Objects tracked
    - top k individuals linked to the top k fitnesses (*) size=(k, )
    - mean individuals at each generation size=(n_generations, dimensionnality)
    """

    def __init__(self, config: int, decoder: Decoder, top_k: int = 3) -> None:
        self.config: dict = config
        self.individuals_dimension: int = decoder.encoding_size()
        self.top_k: int = top_k

    @partial(jit, static_argnums=(0,))
    def init(self) -> TrackerState:
        """Initializes the state of the tracker

        Returns:
            TrackerState: State of the tracker
        """
        return {
            "training": {
                # Fitness of the top k individuals during training (descending order)
                "top_k_fit": jnp.zeros(
                    (self.config["evo"]["n_generations"], self.top_k)
                )
            },
            "eval": {
                # Fitness of the individual at the center of the population,
                # used to draw the offspring population lambda.
                "mean_fit": jnp.zeros((self.config["evo"]["n_generations"],)),
            },
            "backup": {
                # All the mean individuals, (n_gen, indiv_size)
                "sample_mean_ind": jnp.zeros(
                    (
                        self.config["evo"]["n_generations"],
                        self.individuals_dimension,
                    )
                ),
                # genomes of the top k individulas,
                # in descending order according to their fitness
                "top_k_individuals": jnp.zeros(
                    (self.top_k, self.individuals_dimension)
                ),
            },
            "gen": 0,
        }

    @partial(jit, static_argnums=(0, 5))
    def update(
        self,
        tracker_state: TrackerState,
        individuals: chex.Array,
        fitnesses: chex.Array,
        sample_mean: chex.Array,
        eval_f: Callable[[chex.Array, jrd.KeyArray], float],
        rng_eval: jrd.KeyArray,
    ) -> TrackerState:
        """Update the tracker object with the metrics of the current generation"""
        i = tracker_state["gen"]
        # SECTION - top_k fitness and indiv
        # Get top_k fitnesses with carry over and extract top individuals from this

        # Get previously saved fitnesses, size=(top_k) and indiv size=(top_k, d)
        last_fit = (
            tracker_state["training"]["top_k_fit"]
            .at[i - 1]
            .get(mode="fill", fill_value=0.0)
        )
        last_indiv = tracker_state["backup"]["top_k_individuals"]

        if self.config["task"]["maximize"] is True:
            # - merge old fitnesses with new & saved old top_k indiv
            carry_over_fit = jnp.hstack((fitnesses, last_fit))
            carry_over_indiv = jnp.vstack((individuals, last_indiv))
            # - sort index based - jnp.argsort()
            sorted_fitness_idxs = jnp.argsort(carry_over_fit)[::-1]
            # - extract with index best top_k
            top_k_f = carry_over_fit[sorted_fitness_idxs[: self.top_k]]
            top_k_indiv = carry_over_indiv[sorted_fitness_idxs[: self.top_k]]
        else:
            raise ValueError("minimization of the fitness value is not supported")
        # !SECTION - top_k fitness and indivi

        # NOTE - Update top k fitnesses
        tracker_state["training"]["top_k_fit"] = (
            tracker_state["training"]["top_k_fit"].at[i].set(top_k_f)
        )
        # NOTE: Update backup individuals
        tracker_state["backup"]["top_k_individuals"] = top_k_indiv

        tracker_state["backup"]["sample_mean_ind"] = (
            tracker_state["backup"]["sample_mean_ind"].at[i].set(sample_mean)
        )
        # NOTE - Update center of population fitness
        mean_fitness = eval_f(sample_mean, rng_eval)
        tracker_state["eval"]["mean_fit"] = (
            tracker_state["eval"]["mean_fit"].at[i].set(mean_fitness)
        )

        # NOTE - Update current generation counter
        tracker_state["gen"] += 1

        return tracker_state

    def wandb_log(self, tracker_state: TrackerState, wdb_run) -> None:
        gen = tracker_state["gen"] - 1

        wdb_run.log(
            {
                "training": {
                    "top_k_fit": {
                        f"top_{t}_fit": float(
                            tracker_state["training"]["top_k_fit"][gen][t]
                        )
                        for t in range(self.top_k)
                    },
                },
                "eval": {"mean_fit": tracker_state["eval"]["mean_fit"][gen]},
            }
        )

    def wandb_save_genome(
        self,
        genome: chex.Array,
        wdb_run,
        file_name: str = "mean_indiv",
        now: bool = False,
    ) -> None:
        """Saves the current genome to the current wandb run folder
        and uploads the file based on the chosen policy `now`.

        Args:
            genome (chex.Array): Genome to save as a pickled binary file.
            wdb_run (_type_): Current Wandb Run object.
            now (bool, optional):
                if now is false, the upload will be delayed until the end of the run.
                Defaults to False.
        """
        save_path = Path(wdb_run.dir) / "genomes" / file_name
        save_path = save_path.with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            jnp.save(f, genome)

        if now:
            wdb_run.save(str(save_path), base_path=f"{wdb_run.dir}/", policy="now")


def batch_wandb_log(
    wdb_run, statistics, batch_size: int, prefix: str = "individual"
) -> None:
    """Map over each leaf of the list pytrees and logs item in the lists.

    Args:
        wdb_run (_type_): current run
        statistics (_type_): Statistics dictionnary, array of one value per individual.
        batch_size (int): _description_
        prefix (str, optional): _description_. Defaults to "individual".
    """
    log_dict = {
        f"{prefix}_{i}": tree_util.tree_map(lambda e: e[i], statistics)
        for i in range(batch_size)
    }

    wdb_run.log(log_dict)
