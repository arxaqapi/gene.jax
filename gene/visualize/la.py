"""
Perform [Fitness|Loss] Landscape Analysis on a specific run.

"""
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap, Array

from gene.v1.lla import interpolate_2D, plot_ll
from gene.core.evaluation import get_braxv1_env


def run_lla_brax(
    config: dict,
    plot_title: str,
    initial_genome: Array,
    final_genome: Array,
    wandb_run,  # only used to upload files to wandb
    n: int = 100,
) -> None:
    rng, interpolation_rng, eval_rng = jrd.split(rng, 3)

    # NOTE - 1. Get env
    # FIXME - pass it down ?
    env = get_braxv1_env(config)

    # NOTE - 2. get genomes
    # FIXME - (as parameters)
    initial_genome, final_genome = initial_genome, final_genome

    # NOTE - 3. interpolate
    genomes, xs, ys = interpolate_2D(
        initial_genome, final_genome, n=n, key=interpolation_rng
    )

    # NOTE - 4. evaluate at each interpolation step
    # FIXME - eval_func to fix, or to pass down?
    part_eval = partial(evaluate_individual_brax, config=config, rng=eval_rng, env=env)
    vmap_eval = jit(vmap(part_eval, in_axes=(0)))

    values = vmap_eval(genomes)

    # NOTE - 5. plot landscape
    # FIXME - fix directory
    plot_save_path = Path(run.dir) / "lla" / title
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_ll(
        values,
        xs,
        ys,
        evaluate_individual_brax(initial_genome, eval_rng, config, env=env),
        evaluate_individual_brax(final_genome, eval_rng, config, env=env),
        plot_title=str(plot_save_path),
    )

    # NOTE - 6. Save LL to run
    # TODO - test me
    wandb_run.upload_file(f"{str(plot_save_path)}.html", root=f"{wandb_run.dir}")
    wandb_run.upload_file(f"{str(plot_save_path)}.png", root=f"{wandb_run.dir}")
