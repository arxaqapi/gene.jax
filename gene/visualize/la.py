"""
Perform [Fitness|Loss] Landscape Analysis on a specific run.

"""
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap, Array

from gene.core.evaluation import get_braxv1_env
from gene.learning import brax_eval
from gene.core.decoding import Decoder


def interpolate_2D(
    initial_genome: Array,
    final_genome: Array,
    key: jrd.KeyArray,
    n: int = 200,
) -> tuple[Array, Array, Array]:
    """Performs 2D interpolation between the initial and final network.

    - [An empirical analysis of the optimization of deep network loss surfaces](http://arxiv.org/abs/1612.04010)
    - [Visualizing the Loss Landscape of Neural Nets](http://arxiv.org/abs/1712.09913)


    Args:
        initial_genome (jax.Array):
            The genome of the initial individual's neural network.
        final_genome (jax.Array): The genome of the final individual's neural network.
        n (int): number of points to interpolate per axis.
        key (jrd.KeyArray): key used to generate the random vector.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: genomes, x, y
    """
    # raise NotImplementedError
    # taken from https://github.com/TemplierPaul/QDax/blob/main/analysis/landscape_analysis_2d.py
    v1 = final_genome - initial_genome

    # NOTE - Gram-Schmidt Process ??
    # https://www.ucl.ac.uk/~ucahmdl/LessonPlans/Lesson10.pdf
    v2 = jrd.normal(key, shape=v1.shape)
    v2 = v2 - jnp.dot(v2, v1) * v1 / jnp.dot(v1, v1)
    v2 = v2 / jnp.linalg.norm(v2) * jnp.linalg.norm(v1)

    # x, y = jnp.meshgrid(jnp.linspace(-4, 5, n), jnp.linspace(-4, 5, n))
    # x, y = jnp.meshgrid(jnp.linspace(-2, 3, n), jnp.linspace(-2, 3, n))
    x, y = jnp.meshgrid(jnp.linspace(-1, 2, n), jnp.linspace(-1, 2, n))
    _x = x.reshape((-1, 1))
    _y = y.reshape((-1, 1))

    # See equation 1 in http://arxiv.org/abs/1712.09913
    genomes = _x * v1 + _y * v2 + initial_genome
    return genomes, x, y


def plot_ll(
    values: Array,
    X: Array,
    Y: Array,
    initial_genome_fitness: float,
    final_genome_fitness: float,
    title: str,
) -> None:
    """Loss Landscape plotting function.
    Exports the final plot as a png and an interactive html file using plotly.

    Args:
        values (jax.Array): Computed fitness values for each interpolated genomes
        X (jax.Array): X values of the interpolated genomes
        Y (jax.Array): Y values of the interpolated genomes
        initial_genome (jax.Array):
            genomes of the initial individual. Starting point of the search.
        final_genome (jax.Array):
            genomes of the final individual. End point of the search.
        export_name (str, optional):
            Name of the output visualization files. Defaults to "test".
    """
    import plotly.graph_objects as go

    assert X.shape == Y.shape
    values = jnp.array(values).reshape(*X.shape)

    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=values)])
    # Add contour plots
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )

    fig.add_scatter3d(
        name="Initial genome", x=(0,), y=(0,), z=(initial_genome_fitness,), legendrank=1
    )
    # TODO - check y
    fig.add_scatter3d(
        name="Final genome", x=(1,), y=(0,), z=(final_genome_fitness,), legendrank=1
    )

    fig.update_layout(
        title=title,
        # https://plotly.com/python/3d-camera-controls/
        scene_camera={
            "up": dict(x=0, y=0, z=1),
            "center": dict(x=0, y=0, z=0),
            "eye": dict(x=-1.5, y=1.5, z=1.5),
        },
        # https://stackoverflow.com/questions/61827165/plotly-how-to-handle-overlapping-colorbar-and-legends
        # legend_orientation="h",
        # https://plotly.com/python/legend/
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(f"{title}.html")
    fig.write_image(f"{title}.png")


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
