import time
import wandb
import argparse

from gene.meta import meta_learn_cgp_corrected
from gene.utils import fail_if_not_device, load_config

from cgpax.jax_functions import available_functions


if __name__ == "__main__":
    fail_if_not_device()

    parser = argparse.ArgumentParser(
        prog="",
        description="Runs a meta evolution loop and evaluates \
        the best learned distance function.",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="cgp-gene",
        help="Name of the weights and biases project",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="meta-cgp-corrected-long-df",
        help="Name of the w&b run",
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default="arxaqapi",
        help="User used to log on weights and biases.",
    )
    parser.add_argument(
        "-t",
        "--tags",
        nargs="*",
        default=[str(int(time.time()))],
        help="A list of tags for weights and biases",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/cgp_meta_df_corrected_long.json",
        help="Config file used for the meta-evolution and basic evolution tasks.",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=1 / 3,
        help="Beta value used for the fitness shaping. A value of 1 means only the \
            network properties are evaluated",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Seed used for the run",
    )

    args = parser.parse_args()
    meta_config = load_config(args.config)

    meta_config["seed"] = args.seed

    meta_config["cgp_config"] = {
        "seed": 8643,
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

    wandb_run = wandb.init(
        project=args.project,
        config=meta_config,
        tags=args.tags,
        name=args.name,
        entity=args.entity,
    )
    meta_learn_cgp_corrected(meta_config, wandb_run, beta=args.beta)
