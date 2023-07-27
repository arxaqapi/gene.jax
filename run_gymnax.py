import argparse

import wandb
import jax.random as jrd

from gene.learning import learn_gymnax_task
from gene.utils import fail_if_not_device, load_config, validate_json
from gene.core.distances import get_df


if __name__ == "__main__":
    fail_if_not_device()

    parser = argparse.ArgumentParser(
        prog="Gymnax running script",
        description="Performs policy search using an evolutionnary strategy.",
    )

    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="cartpole",
        choices=["cartpole", "acrobot", "pendulum"],
        help="Name of the environnment",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="Gymnax benchmarks",
        help="Name of the wandb project",
    )

    args = parser.parse_args()

    config = load_config(f"config/{args.env}.json")
    validate_json(config)

    df = get_df(config)()

    wdb_run = wandb.init(
        project=args.project,
        config=config,
        name=f"{args.env}-{config['encoding']['distance']}",
    )

    learn_gymnax_task(df, jrd.PRNGKey(0), config, wandb)
