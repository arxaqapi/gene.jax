import argparse
import time
import wandb
from copy import deepcopy

from gene.utils import fail_if_not_device, load_config
from gene.experiment import comparison_experiment_cgp
from gene.meta import meta_learn_cgp_corrected
from cgpax.jax_functions import available_functions


if __name__ == "__main__":
    fail_if_not_device()

    parser = argparse.ArgumentParser(
        prog="Runs a meta evolution loop and evaluates the best learned distance function.",
        description="",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="devnull",
        help="Name of the weights and biases project",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/cgp_meta_df_corrected_w2d.json",
        help="Config file used for the meta-evolution and basic evolution tasks.",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=1.0,
        help="Beta value used for the fitness shaping. A value of 1 means only the network properties are evaluated",
    )

    args = parser.parse_args()

    expe_time = int(time.time())
    extra_tags = ["full-tanh-test"]

    # NOTE - 1. Meta cgp
    meta_config = load_config(args.config)
    meta_config["cgp_config"] = {
        "seed": 3663398,
        "problem": "cgp_meta_df",
        "solver": "cgp",
        "mutation": "standard",
        "n_generations": meta_config["evo"]["n_generations"],
        "n_individuals": meta_config["evo"]["population_size"],
        "p_mut_inputs": 0.15,
        "p_mut_functions": 0.15,
        "p_mut_outputs": 0.3,
        "n_nodes": 64,
        "n_functions": len(available_functions),
        "nan_replacement": 0.0,
        "survival": "truncation",
        "selection": {
            "type": "tournament",
            "elite_size": 8,
            "tour_size": 7,
        },
    }
    meta_config["group"] = "meta"

    with wandb.init(
        project=args.project,
        name="CC-Comp-meta",
        config=meta_config,
        tags=[f"{expe_time}"] + extra_tags,
    ) as meta_cgp_wdb:
        best_cgp_genome_archive = meta_learn_cgp_corrected(
            meta_config=meta_config, wandb_run=meta_cgp_wdb, beta=1
        )

    comparison_experiment_cgp(
        config=deepcopy(next(iter(meta_config["curriculum"].values()))),
        cgp_config=meta_config["cgp_config"],
        cgp_df_genome_archive=best_cgp_genome_archive,
        project=args.project,
        expe_time=expe_time,
        extra_tags=extra_tags,
    )
