"""
Loss Landscape Analysis code.

- Create the grid to interpolate onto.
- Vmap over all points and evaluate interpolated genomes
"""
import jax
import jax.numpy as jnp
import jax.random as jrd

from pathlib import Path

from evaluate import evaluate_individual


def load_genomes(path_initial: Path, path_final: Path) -> tuple[jax.Array, jax.Array]:
    with path_initial.open("rb") as f:
        initial_genome = jnp.load(f)
    with path_final.open("rb") as f:
        final_genome = jnp.load(f)
    return initial_genome, final_genome


def evaluate_all(
    genomes: list[jax.Array], rng: jrd.KeyArray, config: dict
) -> list[float]:
    """Evaluate all individual in the list of genomes

    Args:
        genomes (list[jax.Array]): list of the genomes of all individuals to evaluate
        rng (jrd.KeyArray): rng key used to run the simulation
        config (dict): config of the current run

    Returns:
        list[float]: List of floats corresponding to the fitness value per individual
    """
    return [evaluate_individual(genome, rng, config) for genome in genomes]


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
    initial_genome: jax.Array,
    final_genome: jax.Array,
    export_name: str = "test",
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

    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=values)])

    # FIXME: add the correct values for initial_genome and final_genome / extract from `values`
    x_initial, y_initial, z_initial = *initial_genome, 1
    x_final, y_final, z_final = *final_genome, 10
    fig.add_scatter3d(
        name="Initial genome",
        x=(x_initial,),
        y=(y_initial,),
        z=(z_initial,),
        legendrank=1,
    )
    fig.add_scatter3d(
        name="Final genome", x=(x_final,), y=(y_final,), z=(z_final,), legendrank=1
    )

    fig.update_layout(
        title="3D plot test",
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

    fig.write_html(f"{export_name}.html")
    fig.write_image(f"{export_name}.png")

# TODO: finish me
def lla(config: dict, rng: jrd.KeyArray = jrd.PRNGKey(0)):
    rng, interpolation_rng, eval_rng = jrd.split(rng, 3)

    # SECTION - downlad files from run
    # https://docs.wandb.ai/guides/track/public-api-guide#download-a-file-from-a-run
    import wandb 
    api = wandb.Api()
    run = api.run("<entity>/<project>/<run_id>")
    path_initial = run.file("...").download()
    path_final = run.file("...").download()
    # !SECTION

    # 1. load files
    initial_genome, final_genome = load_genomes(
        path_initial,
        path_final)
    # 2. interpolate
    genomes, xs, ys = interpolate_2D(
        initial_genome, final_genome, n=10, key=interpolation_rng
    )
    # 3. evaluate at each interpolation step
    evaluate_all(genomes, rng=eval_rng, config=config)
    # 4. plot landscape
    plot_ll(genomes, xs, ys, initial_genome, final_genome)


if __name__ == "__main__":
    lla(None, None)
