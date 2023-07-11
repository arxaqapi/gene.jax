"""
Perform [Fitness|Loss] Landscape Analysis on a specific run.

"""
from functools import partial
from pathlib import Path

import jax.random as jrd
from jax import jit, vmap, Array

# FIXME - to remove
from gene.v1.lla import interpolate_2D, plot_ll
from gene.core.evaluation import get_braxv1_env
from gene.learning import brax_eval
from gene.core.decoding import Decoder


def run_fla_brax(
    plot_title: str,
    config: dict,
    initial_genome: Array,
    final_genome: Array,
    decoder: Decoder,
    wdb_run,  # only used to upload files to wandb
    n: int = 100,
    rng: jrd.KeyArray = jrd.PRNGKey(0),
) -> None:
    """Runs 2D Fitness Landscape Analysis between an `initial_genome`
    and a `final_genome`. Saves the plot on weights and biases


    Args:
        config (dict): config file of the run
        plot_title (str): title given to the output plot
        initial_genome (Array): Initial genome used to interpolate the landscape
        final_genome (Array): Final genome used to interpolate the landscape
        wdb_run (_type_): Run object used to upload the plots.
    """
    rng, interpolation_rng, eval_rng = jrd.split(rng, 3)

    # NOTE - 1. Get env
    env = get_braxv1_env(config)

    # NOTE - 2. get genomes
    initial_genome, final_genome = initial_genome, final_genome

    # NOTE - 3. interpolate
    genomes, xs, ys = interpolate_2D(
        initial_genome, final_genome, n=n, key=interpolation_rng
    )

    # NOTE - 4. evaluate at each interpolation step
    part_eval = partial(
        brax_eval, rng=eval_rng, decoder=decoder, config=config, env=env
    )
    vmap_eval = jit(vmap(part_eval, in_axes=(0)))

    values = vmap_eval(genomes)

    # NOTE - 5. plot landscape
    plot_save_path = Path(wdb_run.dir) / "lla" / plot_title
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto sync at end of run
    plot_ll(
        values,
        xs,
        ys,
        brax_eval(initial_genome, eval_rng, decoder, config, env=env),
        brax_eval(final_genome, eval_rng, decoder, config, env=env),
        title=str(plot_save_path),
    )
