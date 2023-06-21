from jax import jit, vmap
import jax.random as jrd
import wandb

from gene.lla import load_genomes, interpolate_2D, plot_ll
from gene.evaluate import evaluate_individual, evaluate_individual_brax, get_brax_env

from functools import partial
from pathlib import Path


def run_lla_gymnax(
    rng,
    run_name: str = "arxaqapi/Cartpole/qvobnkry",  # direct | seed 9
    initial_genome_name: str = "genomes/1685094639_g0_mean_indiv.npy",
    final_genome_name: str = "genomes/1685094639_g100_mean_indiv.npy",
    title: str = "",
    n: int = 100,
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
        initial_genome, final_genome, n=n, key=interpolation_rng
    )

    # NOTE - 4. evaluate at each interpolation step
    part_eval = partial(evaluate_individual, config=config, rng=eval_rng)
    vmap_eval = jit(vmap(part_eval, in_axes=(0)))

    values = vmap_eval(genomes)

    # NOTE - 5. plot landscape
    plot_save_path = Path(run.dir) / "lla" / title
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_ll(
        values,
        xs,
        ys,
        evaluate_individual(initial_genome, eval_rng, config),
        evaluate_individual(final_genome, eval_rng, config),
        title=str(plot_save_path),
    )

    # NOTE - 6. Save LL to run
    run.upload_file(f"{str(plot_save_path)}.html", root=f"{run.dir}")
    run.upload_file(f"{str(plot_save_path)}.png", root=f"{run.dir}")


def run_lla_brax(
    rng: jrd.KeyArray,
    run_name: str,
    initial_genome_name: str,
    final_genome_name: str,
    title: str,
    n: int = 100,
) -> None:
    rng, interpolation_rng, eval_rng = jrd.split(rng, 3)

    # NOTE - 1.1. download files from run
    api = wandb.Api()
    run = api.run(run_name)
    config = run.config

    path_initial = run.file(initial_genome_name).download(replace=True).name
    path_final = run.file(final_genome_name).download(replace=True).name

    # NOTE - 1.2. Get env
    env = get_brax_env(config)

    # NOTE - 2. load files
    initial_genome, final_genome = load_genomes(path_initial, path_final)

    # NOTE - 3. interpolate
    genomes, xs, ys = interpolate_2D(
        initial_genome, final_genome, n=n, key=interpolation_rng
    )

    # NOTE - 4. evaluate at each interpolation step
    part_eval = partial(evaluate_individual_brax, config=config, rng=eval_rng, env=env)
    vmap_eval = jit(vmap(part_eval, in_axes=(0)))

    values = vmap_eval(genomes)

    # NOTE - 5. plot landscape
    plot_save_path = Path(run.dir) / "lla" / title
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_ll(
        values,
        xs,
        ys,
        evaluate_individual_brax(initial_genome, eval_rng, config, env=env),
        evaluate_individual_brax(final_genome, eval_rng, config, env=env),
        title=str(plot_save_path),
    )

    # NOTE - 6. Save LL to run
    run.upload_file(f"{str(plot_save_path)}.html", root=f"{run.dir}")
    run.upload_file(f"{str(plot_save_path)}.png", root=f"{run.dir}")


if __name__ == "__main__":
    rng = jrd.PRNGKey(1)

    # run_lla_gymnax(
    #     rng,
    #     run_name="arxaqapi/Cartpole/r1jh45f7",
    #     initial_genome_name="genomes/1685094261_g0_mean_indiv.npy",
    #     final_genome_name="genomes/1685094261_g100_mean_indiv.npy",
    #     title="GENE encoding on Cartpole",
    # )

    run_lla_brax(
        rng,
        run_name="arxaqapi/Brax halfcheetah/qugsbu6j",
        initial_genome_name="genomes/g000_mean_indiv.npy",
        final_genome_name="genomes/g500_mean_indiv.npy",
        title="GENE encoding on Brax Halfcheetah",
    )
