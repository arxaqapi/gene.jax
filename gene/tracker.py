from functools import partial
from pathlib import Path

import jax.numpy as jnp
from jax import jit
import chex

from gene.encoding import Encoding_size_function


# NOTE: once initialized, the object should not be modified in compiled functions
# TODO: add plot possibility (error bars & stuff)
class Tracker:
    def __init__(self, config: dict, top_k: int = 3) -> None:
        self.config: dict = config
        self.top_k: int = top_k
        self.genome_counter: int = 0

    @partial(jit, static_argnums=(0,))
    def init(self) -> chex.ArrayTree:
        """Initialize the tracker state where:

        Returns:
            chex.ArrayTree: State of the tracker
        """
        return {
            "training": {
                # Fitness of the top k individuals during training (ordered)
                "top_k_fit": jnp.zeros(
                    (self.config["evo"]["n_generations"], self.top_k)
                ),
                # Empirical mean of the fitness of the complete population
                "empirical_mean_fit": jnp.zeros((self.config["evo"]["n_generations"],)),
                # Standart deviation of the fitness of the complete population
                "empirical_mean_std": jnp.zeros((self.config["evo"]["n_generations"],)),
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
                        Encoding_size_function[self.config["encoding"]["type"]](
                            self.config
                        ),
                    )
                ),
            },
            "gen": 0,
        }

    @partial(jit, static_argnums=(0, 4))
    def update(
        self,
        tracker_state: chex.ArrayTree,
        fitness: chex.Array,
        mean_ind: chex.Array,
        eval_f,
        rng_eval,
    ) -> chex.ArrayTree:
        """Update the tracker object with the metrics of the current generation

        Args:
            tracker_state (chex.ArrayTree): _description_
            fitness (chex.Array): _description_
            mean_ind (chex.Array): _description_
            eval_f (_type_): _description_
            rng_eval (_type_): _description_

        Returns:
            chex.ArrayTree: _description_
        """
        i = tracker_state["gen"]
        # [Training] - update top_k_fitness using old state (carry best over)
        last_fit = (
            tracker_state["training"]["top_k_fit"]
            .at[i - 1]
            .get(mode="fill", fill_value=0.0)
        )
        # TODO - argmax/armgin | maximize/minimize
        top_k_f = jnp.sort(jnp.hstack((fitness, last_fit)))[::-1][: self.top_k]

        # NOTE - Update top k fitnesses
        tracker_state["training"]["top_k_fit"] = (
            tracker_state["training"]["top_k_fit"].at[i].set(top_k_f)
        )
        # NOTE - Update empirical fitness mean and std
        tracker_state["training"]["empirical_mean_fit"] = (
            tracker_state["training"]["empirical_mean_fit"].at[i].set(fitness.mean())
        )
        tracker_state["training"]["empirical_mean_std"] = (
            tracker_state["training"]["empirical_mean_std"].at[i].set(fitness.std())
        )

        # NOTE - Update center of population fitness
        fitness = eval_f(mean_ind, rng_eval)
        tracker_state["eval"]["mean_fit"] = (
            tracker_state["eval"]["mean_fit"].at[i].set(fitness)
        )

        # NOTE: Update backup individuals
        tracker_state["backup"]["sample_mean_ind"] = (
            tracker_state["backup"]["sample_mean_ind"].at[i].set(mean_ind)
        )

        # NOTE - Update current generation counter
        tracker_state["gen"] += 1
        return tracker_state

    def wandb_log(self, tracker_state, wdb_run) -> None:
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
                    "empirical_mean_fit": float(
                        tracker_state["training"]["empirical_mean_fit"][gen]
                    ),
                    "empirical_mean_std": float(
                        tracker_state["training"]["empirical_mean_std"][gen]
                    ),
                },
                "eval": {"mean_fit": tracker_state["eval"]["mean_fit"][gen]},
            }
        )

    def wandb_save_genome(self, genome: chex.Array, wdb_run, now: bool = False) -> None:
        gen_string = f"g{str(self.genome_counter).zfill(3)}_"
        save_path = Path(wdb_run.dir) / "genomes" / f"{gen_string}mean_indiv.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            jnp.save(f, genome)

        if now:
            wdb_run.save(str(save_path), base_path=f"{wdb_run.dir}/", policy="now")

        self.genome_counter += 1
