from pathlib import Path
import argparse

import jax.numpy as jnp
import jax.random as jrd
import wandb

from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.core.models import BoundedLinearModelConf
from gene.core.distances import Distance_functions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Brax Visualization",
        description="Performs the evaluation of a learned policy genome by \
            retrieving it from weights and biases.",
    )

    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="Brax expe bench test",
        help="Name of the wandb project",
    )
    parser.add_argument(
        "-r",
        "--run_id",
        type=str,
        default="6k8oolw0",
        help="identifier of the wandb run.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="final_best_indiv.npy",
        help="File to retrieve from wandb previously defined run.",
    )
    parser.add_argument(
        "--rng",
        type=int,
        default=0,
        help="RNG key used to re-evaluate the genome.",
    )

    args = parser.parse_args()

    run_id = f"arxaqapi/{args.project}/{args.run_id}"
    genome_id = args.file

    api = wandb.Api()
    run = api.run(run_id)
    config = run.config

    with open(run.file(f"genomes/{genome_id}").download(replace=True).name, "rb") as f:
        genome = jnp.load(f)

    path = Path("html")
    path.mkdir(exist_ok=True)

    render_brax(
        *visualize_brax(
            config,
            genome,
            model=BoundedLinearModelConf(config),
            df=Distance_functions[config["encoding"]["distance"]](),
            rng=jrd.PRNGKey(args.rng),
        ),
        path / f"run_{run_id.split('/')[-1]}_{genome_id[:-4]}",
    )
