from functools import partial
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import jax.random as jrd
from jax import jit, tree_util, Array
import chex

from gene.core.decoding import Decoder

TrackerState = dict


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

    def __init__(self, config: dict, decoder: Decoder, top_k: int = 3) -> None:
        self.config: dict = config
        self.individuals_dimension: int = decoder.encoding_size()
        self.top_k: int = top_k

    @partial(jit, static_argnums=(0, 1))
    def init(self, skip_mean_backup: bool = False) -> TrackerState:
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
                "sample_mean_ind": None
                if skip_mean_backup
                else jnp.zeros(
                    (
                        self.config["evo"]["n_generations"],
                        self.individuals_dimension,
                    )
                ),
                "initial_mean_indiv": jnp.zeros((self.individuals_dimension,)),
                "final_mean_indiv": jnp.zeros((self.individuals_dimension,)),
                # genomes of the top k individulas,
                # in descending order according to their fitness
                "top_k_individuals": jnp.zeros(
                    (self.top_k, self.individuals_dimension)
                ),
            },
            "gen": 0,
        }

    @partial(jit, static_argnums=(0, 5, 7))
    def update(
        self,
        tracker_state: TrackerState,
        individuals: chex.Array,
        fitnesses: chex.Array,
        sample_mean: chex.Array,
        eval_f: Callable[[chex.Array, jrd.KeyArray], float],
        rng_eval: jrd.KeyArray,
        skip_mean_backup: bool = False,
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

        if not skip_mean_backup:
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

    def wandb_log(self, tracker_state: TrackerState, wdb_run, extra: dict = {}) -> None:
        gen = tracker_state["gen"] - 1

        to_log = {
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
        for k, v in extra.items():
            to_log["training"][k] = v

        wdb_run.log(to_log)

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

    # specific setters
    def set_initial_mean(
        self, tracker_state: TrackerState, indiv: Array
    ) -> TrackerState:
        tracker_state["backup"]["initial_mean_indiv"] = indiv
        return tracker_state

    def set_final_mean(self, tracker_state: TrackerState, indiv: Array) -> TrackerState:
        tracker_state["backup"]["final_mean_indiv"] = indiv
        return tracker_state

    # specific getters
    def get_initial_center_individual(self, tracker_state: TrackerState) -> Array:
        return tracker_state["backup"]["initial_mean_indiv"]

    def get_final_center_individual(self, tracker_state: TrackerState) -> Array:
        return tracker_state["backup"]["final_mean_indiv"]

    def get_top_k_genomes(self, tracker_state) -> Array:
        """Returns the k top individuals"""
        return tracker_state["backup"]["top_k_individuals"]

    def get_mean_fitnesses(self, tracker_state) -> Array:
        return tracker_state["eval"]["mean_fit"]


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


class MetaTracker:
    """Keeps track of:
    - all dfs encountered during training & last sample mean
    - stats:
        - cp max fitness
        - cp mean/var fitness
        - hc 100 & 1000 max fitness
        - hc 100 & 1000 mean/var fitness
        - normalized fitness /
    """

    def __init__(self, config: dict, decoder: Decoder, top_k: int = 3) -> None:
        self.config: dict = config
        self.individuals_dimension: int = decoder.encoding_size()
        self.top_k: int = top_k

    @partial(jit, static_argnums=(0,))
    def init(self) -> TrackerState:
        return {
            "training": {
                "total_emp_mean_fitness": jnp.zeros(
                    (self.config["evo"]["n_generations"])
                ),
                "total_max_fitness": jnp.zeros((self.config["evo"]["n_generations"])),
                "cart": {
                    # max fitness obtained at each generation
                    "max_fitness": jnp.zeros((self.config["evo"]["n_generations"])),
                    "emp_mean_fitnesses": jnp.zeros(
                        (self.config["evo"]["n_generations"])
                    ),
                },
                "hc_100": {
                    "max_fitness": jnp.zeros((self.config["evo"]["n_generations"])),
                    "emp_mean_fitnesses": jnp.zeros(
                        (self.config["evo"]["n_generations"])
                    ),
                },
                "hc_1000": {
                    "max_fitness": jnp.zeros((self.config["evo"]["n_generations"])),
                    "emp_mean_fitnesses": jnp.zeros(
                        (self.config["evo"]["n_generations"])
                    ),
                },
            },
            "backup": {
                "max_df": jnp.zeros(
                    (
                        self.config["evo"]["n_generations"],
                        self.individuals_dimension,
                    )
                ),
                "mean_df": jnp.zeros(
                    (
                        self.config["evo"]["n_generations"],
                        self.individuals_dimension,
                    )
                ),
            },
            "gen": 0,
        }

    def update(
        self,
        tracker_state: TrackerState,
        fitness_value: dict,
        max_df: Array,
        mean_df: Array,
        gen: int = 0,
    ) -> TrackerState:
        # NOTE - Fitness values
        # overall
        tracker_state["training"]["total_emp_mean_fitness"] = (
            tracker_state["training"]["total_emp_mean_fitness"]
            .at[gen]
            .set(fitness_value["total_emp_mean_fitness"])
        )
        # update max fitness if better than previous
        if (
            tracker_state["training"]["total_max_fitness"][gen - 1]
            < fitness_value["total_max_fitness"]
        ):
            tracker_state["training"]["total_max_fitness"] = (
                tracker_state["training"]["total_max_fitness"]
                .at[gen]
                .set(fitness_value["total_max_fitness"])
            )
        # cartpole
        tracker_state["training"]["cart"]["max_fitness"] = (
            tracker_state["training"]["cart"]["max_fitness"]
            .at[gen]
            .set(fitness_value["cart"]["max_fitness"])
        )
        tracker_state["training"]["cart"]["emp_mean_fitnesses"] = (
            tracker_state["training"]["cart"]["emp_mean_fitnesses"]
            .at[gen]
            .set(fitness_value["cart"]["emp_mean_fitnesses"])
        )
        # hc_100
        tracker_state["training"]["hc_100"]["max_fitness"] = (
            tracker_state["training"]["hc_100"]["max_fitness"]
            .at[gen]
            .set(fitness_value["hc_100"]["max_fitness"])
        )
        tracker_state["training"]["hc_100"]["emp_mean_fitnesses"] = (
            tracker_state["training"]["hc_100"]["emp_mean_fitnesses"]
            .at[gen]
            .set(fitness_value["hc_100"]["emp_mean_fitnesses"])
        )
        # hc_1000
        tracker_state["training"]["hc_1000"]["max_fitness"] = (
            tracker_state["training"]["hc_1000"]["max_fitness"]
            .at[gen]
            .set(fitness_value["hc_1000"]["max_fitness"])
        )
        tracker_state["training"]["hc_1000"]["emp_mean_fitnesses"] = (
            tracker_state["training"]["hc_1000"]["emp_mean_fitnesses"]
            .at[gen]
            .set(fitness_value["hc_1000"]["emp_mean_fitnesses"])
        )
        # NOTE - Backup
        tracker_state["backup"]["max_df"] = (
            tracker_state["backup"]["max_df"].at[gen].set(max_df)
        )
        tracker_state["backup"]["mean_df"] = (
            tracker_state["backup"]["mean_df"].at[gen].set(mean_df)
        )

        tracker_state["gen"] += 1

        return tracker_state

    def wandb_log(self, tracker_state: TrackerState, wdb_run) -> None:
        # gen - 1 because this is run after tracker.update(...)
        gen = tracker_state["gen"] - 1

        wdb_run.log(
            {
                "training": {
                    "total_emp_mean_fitness": tracker_state["training"][
                        "total_emp_mean_fitness"
                    ][gen],
                    # TODO
                    "total_max_fitness": tracker_state["training"]["total_max_fitness"][
                        gen
                    ],
                    "cart": {
                        # max fitness obtained at each generation
                        "max_fitness": tracker_state["training"]["cart"]["max_fitness"][
                            gen
                        ],
                        "emp_mean_fitnesses": tracker_state["training"]["cart"][
                            "emp_mean_fitnesses"
                        ][gen],
                    },
                    "hc_100": {
                        "max_fitness": tracker_state["training"]["hc_100"][
                            "max_fitness"
                        ][gen],
                        "emp_mean_fitnesses": tracker_state["training"]["hc_100"][
                            "emp_mean_fitnesses"
                        ][gen],
                    },
                    "hc_1000": {
                        "max_fitness": tracker_state["training"]["hc_1000"][
                            "max_fitness"
                        ][gen],
                        "emp_mean_fitnesses": tracker_state["training"]["hc_1000"][
                            "emp_mean_fitnesses"
                        ][gen],
                    },
                }
            }
        )
        # tree_util.map(lambda e: e[gen], tracker_state["training"])

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
        save_path = Path(wdb_run.dir) / "df_genomes" / file_name
        save_path = save_path.with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            jnp.save(f, genome)

        if now:
            wdb_run.save(str(save_path), base_path=f"{wdb_run.dir}/", policy="now")
