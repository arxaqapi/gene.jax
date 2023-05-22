"""
Loss Landscape Analysis code.

- Create the grid to interpolate onto.
- Vmap over all points and evaluate interpolated genomes
"""
import jax
import jax.numpy as jnp
import jax.random as jrd


def load_genomes() -> tuple[jax.Array, jax.Array]:
    pass


def interpolate_2D(
    initial_genome: jax.Array,
    final_genome: jax.Array,
    n: int = 100,
    key: jrd.KeyArray = jrd.PRNGKey(0),
):
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


def plot(values, X, Y, initial_genome, final_genome, export_name: str = "test"):
    import plotly.graph_objects as go

    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=(X / 2 + Y / 2) ** 2)])

    fig.add_scatter3d(
        name="Initial genome", text="hein?", x=(0,), y=(0,), z=(0,), legendrank=1
    )
    fig.add_scatter3d(name="Final genome", x=(1,), y=(1,), z=(3,), legendrank=1)

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

    # fig.show()
    fig.write_html(f"{export_name}.html")
    fig.write_image(f"{export_name}.png")


def _plot(values: jax.Array, x, y, initial_genome, final_genome):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection="3d")

    # ax.set_zlim(-2, 5)

    # Data
    X = x
    Y = y
    Z = X + Y

    # Plot the surface.  | cm.RdYlGn, cm.viridis, cm.coolwarm
    surf = ax.plot_surface(
        X, Y, Z, cmap=cm.viridis, linewidth=1, antialiased=False, alpha=0.6
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Plot the points of interests
    # ax.scatter(initial_genome[0], initial_genome[1], 0, c="black", s=100, alpha=1)
    # ax.scatter(final_genome[0], final_genome[1], 1 * 2+0.5, c="green", s=100, alpha=1)

    ax.stem(
        [initial_genome[0], final_genome[0]],
        [initial_genome[1], final_genome[1]],
        [1, 3],
        bottom=-2,
    )
    # ax.stem(
    #     [initial_genome[0], final_genome[0]],
    #     [initial_genome[1], final_genome[1]],
    #     [0, 1])

    # Add a color bar which maps values to colors.
    fig.savefig("test.png")


if __name__ == "__main__":
    initial_genome = jnp.array([0, 0])
    final_genome = jnp.array([1, 1])

    genomes, x, y = interpolate_2D(initial_genome, final_genome, n=20)

    plot(genomes, x, y, initial_genome, final_genome)
