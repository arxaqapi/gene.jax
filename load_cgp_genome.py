import wandb
import jax.numpy as jnp
from cgpax.analysis.genome_analysis import __save_graph__, __write_readable_program__


def load_run(run_id: str):
    api = wandb.Api()
    run = api.run(run_id)
    return run

def get_file(filepath: str, run):
    with open(run.file(filepath).download(replace=True).name, "rb") as f:
        genome = jnp.load(f)
    return genome
