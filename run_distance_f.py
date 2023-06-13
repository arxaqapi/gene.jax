import wandb
import jax.numpy as jnp

from gene.learn_distance import learn_distance_f_evo


if __name__ == "__main__":
    import json

    with open("config/distance_learning.json") as f:
        config = json.load(f)

    # SECTION - Get learned gene genome
    _old_run = wandb.Api().run("arxaqapi/Brax halfcheetah/r9sy0brc")
    with open(
        _old_run.file("genomes/g996_mean_indiv.npy").download(replace=True).name, "rb"
    ) as f:
        sample_center_genome = jnp.load(f)
    # !SECTION - Get learned gene genome

    wdb_run = wandb.init(
        project="Distance-tests",
        config=config,
    )

    fit, center = learn_distance_f_evo(
        config=config, wdb_run=wdb_run, sample_center_genome=sample_center_genome
    )
