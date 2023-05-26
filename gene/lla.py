"""
Loss Landscape Analysis code.

- Create the grid to interpolate onto.
- Vmap over all points and evaluate interpolated genomes
"""
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap, lax
from brax import envs
from brax.envs.wrapper import EpisodeWrapper

from pathlib import Path
from functools import partial


def get_env(config: dict):
    env = envs.get_environment(env_name=config["problem"]["environnment"])
    return EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )


def load_genomes(path_initial: Path, path_final: Path) -> tuple[jax.Array, jax.Array]:
    with open(path_initial, "rb") as f:
        initial_genome = jnp.load(f)
    with open(path_final, "rb") as f:
        final_genome = jnp.load(f)
    return initial_genome, final_genome


def interpolate_2D(
    initial_genome: jax.Array,
    final_genome: jax.Array,
    n: int,
    key: jrd.KeyArray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Performs 2D interpolation between the initial and final network.

    - .. _An empirical analysis of the optimization of deep network loss surfaces: http://arxiv.org/abs/1612.04010.
    - .. _Visualizing the Loss Landscape of Neural Nets: http://arxiv.org/abs/1712.09913.


    Args:
        initial_genome (jax.Array): The genome of the initial individual's neural network.
        final_genome (jax.Array): The genome of the final individual's neural network.
        n (int): number of points to interpolate per axis.
        key (jrd.KeyArray): key used to generate the random vector.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: genomes, x, y
    """
    # taken from https://github.com/TemplierPaul/QDax/blob/main/analysis/landscape_analysis_2d.py
    v1 = final_genome - initial_genome

    # NOTE - Gram-Schmidt Process ??
    # https://www.ucl.ac.uk/~ucahmdl/LessonPlans/Lesson10.pdf
    v2 = jax.random.normal(key, shape=v1.shape)
    v2 = v2 - jnp.dot(v2, v1) * v1 / jnp.dot(v1, v1)
    v2 = v2 / jnp.linalg.norm(v2) * jnp.linalg.norm(v1)

    x, y = jnp.meshgrid(jnp.linspace(-1, 2, n), jnp.linspace(-1, 2, n))
    _x = x.reshape((-1, 1))
    _y = y.reshape((-1, 1))

    # See equation 1 in http://arxiv.org/abs/1712.09913
    genomes = _x * v1 + _y * v2 + initial_genome
    return genomes, x, y


def plot_ll(
    values: jax.Array,
    X: jax.Array,
    Y: jax.Array,
    initial_genome_fitness: float,
    final_genome_fitness: float,
    title: str,
) -> None:
    """Loss Landscape plotting function. Exports the final plot as a png and an interactive html file using plotly.

    Args:
        values (jax.Array): Computed fitness values for each interpolated genomes
        X (jax.Array): X values of the interpolated genomes
        Y (jax.Array): Y values of the interpolated genomes
        initial_genome (jax.Array): genomes of the initial individual. Starting point of the search.
        final_genome (jax.Array): genomes of the final individual. End point of the search.
        export_name (str, optional): Name of the output visualization files. Defaults to "test".
    """
    import plotly.graph_objects as go

    assert X.shape == Y.shape
    values = jnp.array(values).reshape(*X.shape)

    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=values)])
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

    fig.write_html(f"{title.replace(' ', '_')}.html")
    fig.write_image(f"{title.replace(' ', '_')}.png")
