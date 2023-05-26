from jax import lax, jit, vmap
import jax.numpy as jnp
import jax.random as jrd
import wandb

from gene.lla import load_genomes, interpolate_2D, plot_ll
from gene.evaluate import evaluate_individual

from functools import partial


# FIXME -
def run_lla_gymnax(
    rng,
    run_name: str = "arxaqapi/Cartpole/qvobnkry",  # direct | seed 9
    initial_genome_name: str = "genomes/1685094639_g0_mean_indiv.npy",
    final_genome_name: str = "genomes/1685094639_g100_mean_indiv.npy",
    title: str = "",
):
    rng, interpolation_rng, eval_rng = jrd.split(rng, 3)

    # NOTE - 1. download files from run
    api = wandb.Api()
    run = api.run(run_name)
    config = run.config

    path_initial = run.file(initial_genome_name).download(replace=True).name
    path_final = run.file(final_genome_name).download(replace=True).name

    # NOTE - 2. load files
    initial_genome, final_genome = load_genomes(path_initial, path_final)

    # NOTE - 3. interpolate
    genomes, xs, ys = interpolate_2D(
        initial_genome, final_genome, n=100, key=interpolation_rng
    )

    # NOTE - 4. evaluate at each interpolation step
    #
    part_eval = partial(evaluate_individual, config=config, rng=eval_rng)
    vmap_eval = jit(vmap(part_eval, in_axes=(0)))

    values = vmap_eval(genomes)

    # NOTE - 5. plot landscape
    plot_ll(
        values,
        xs,
        ys,
        evaluate_individual(initial_genome, eval_rng, config),
        evaluate_individual(final_genome, eval_rng, config),
        title=title,
    )


if __name__ == "__main__":
    rng = jrd.PRNGKey(1)

    run_lla_gymnax(rng, title="Direct encoding on Cartpole")

    run_lla_gymnax(
        rng,
        run_name="arxaqapi/Cartpole/r1jh45f7",
        initial_genome_name="genomes/1685094261_g0_mean_indiv.npy",
        final_genome_name="genomes/1685094261_g100_mean_indiv.npy",
        title="GENE encoding on Cartpole",
    )
