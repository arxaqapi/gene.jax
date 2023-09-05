from pathlib import Path
import time


import jax.numpy as jnp
import wandb

from gene.tracker import Tracker
from gene.learning import learn_brax_task
from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.visualize.la import run_fla_brax
from gene.visualize.neurons import visualize_neurons_3d, visualize_neurons_2d
from gene.core.distances import get_df, DistanceFunction, NNDistance
from gene.core.models import get_model
from gene.core.decoding import get_decoder


class Experiment:
    """Perform an experiment composed of a set of runs,
    the corresponding FLA, and the visualization of the result.
    """

    def __init__(self, config: dict, project_name, tags: list[str] = []) -> None:
        self.config = config
        self.project_name = project_name
        self.tags = tags

    def run(self, seed: int, name=None, save_step: int = 2000):
        self.config["seed"] = seed

        wdb_run = wandb.init(
            project=self.project_name, name=name, config=self.config, tags=self.tags
        )

        df = get_df(self.config)()
        tracker, tracker_state = learn_brax_task(
            self.config, df=df, wdb_run=wdb_run, save_step=save_step
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

            neuron_pos_path = Path(wdb_run.dir) / "neurons_positions"
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

        wdb_run.finish()

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


def meta_comparison_experiment(config: dict, project_name: str = "CC bench comparison"):
    """Loads a learned DF and uses it to do policy search on various
    tasks and compare with pL2 and direct encoding."""
    assert config["encoding"]["type"] == "gene"

    timestamp = int(time.time())
    api = wandb.Api()
    run = api.run("arxaqapi/Meta df benchmarks/xt8byi35")
    learned_config = run.config
    with open(
        run.file("df_genomes/mg_1899_best_genome.npy").download(replace=True).name, "rb"
    ) as f:
        df_genome = jnp.load(f)

    learned_df: DistanceFunction | None = NNDistance(
        df_genome, learned_config, learned_config["net"]["layer_dimensions"]
    )

    wdb_run_learned = wandb.init(
        project=project_name,
        name="CC-Bench-comp-learned",
        config=config,
        tags=["learned-df", f"{timestamp}"],
    )

    learn_brax_task(config=config, df=learned_df, wdb_run=wdb_run_learned)
    wdb_run_learned.finish()

    # NOTE - Compare to pL2
    config["encoding"]["distance"] = "pL2"

    wdb_run_pL2 = wandb.init(
        project=project_name,
        name="CC-Bench-comp-pL2",
        config=config,
        tags=["pL2", f"{timestamp}"],
    )
    # get df
    pL2 = get_df(config)()
    learn_brax_task(config=config, df=pL2, wdb_run=wdb_run_pL2)
    wdb_run_pL2.finish()

    # NOTE - Compare to direct encoding
    config["encoding"]["type"] = "direct"

    wdb_run_direct = wandb.init(
        project=project_name,
        name="CC-Bench-comp-direct",
        config=config,
        tags=["direct", f"{timestamp}"],
    )
    learn_brax_task(config=config, df=get_df(config)(), wdb_run=wdb_run_direct)
    wdb_run_direct.finish()
