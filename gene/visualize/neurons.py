from pathlib import Path
from typing import Union

import jax.numpy as jnp
from jax import Array
import plotly.graph_objects as go


def visualize_neurons_2d(genome: Array, config: dict, title: str):
    """Takes a genotype, extracts all neuron position vectors and plots them in 2D space
    Saves the plot as `png` and `html` files

    Args:
        genome (Array): genome to plot
        config (dict): config linked to the genome
        title (str): title of the plot, also used as filename
    """
    layer_dims = config["net"]["layer_dimensions"]
    d = config["encoding"]["d"]

    if d != 2:
        raise ValueError("Encoding dimension is not 2.")

    # Split genome, ditch biases
    genome_w, _ = jnp.split(genome, [sum(layer_dims) * d])
    genome_w_positions = jnp.array(jnp.split(genome_w, sum(layer_dims)))

    # init figure object
    fig = go.Figure()
    # for each layer, get subsection
    for i, layer in enumerate(layer_dims):
        start_i = sum(layer_dims[:i])
        stop_i = layer + sum(layer_dims[:i])

        #  Subplot for each layer
        fig.add_scatter(
            name=f"layer {i}",
            x=genome_w_positions[start_i:stop_i, 0],
            y=genome_w_positions[start_i:stop_i, 1],
            mode="markers",
            legendrank=1,
        )

    # save as png and html
    fig.write_html(f"{title}.html")
    fig.write_image(f"{title}.png")


def visualize_neurons_3d(genome: Array, config: dict, title: Union[str, Path]):
    """Takes a genotype, extracts all neuron position vectors and plots them in 3D space
    Saves the plot as `png` and `html` files

    Args:
        genome (Array): genome to plot
        config (dict): config linked to the genome
        title (str): title of the plot, also used as filename
    """
    layer_dims = config["net"]["layer_dimensions"]
    d = config["encoding"]["d"]

    if d != 3:
        raise ValueError("Encoding dimension is not 3.")

    # Split genome, ditch biases
    genome_w, _ = jnp.split(genome, [sum(layer_dims) * d])
    genome_w_positions = jnp.array(jnp.split(genome_w, sum(layer_dims)))

    # init figure object
    fig = go.Figure()
    # for each layer, get subsection
    for i, layer in enumerate(layer_dims):
        start_i = sum(layer_dims[:i])
        stop_i = layer + sum(layer_dims[:i])

        #  Subplot for each layer
        fig.add_scatter3d(
            name=f"layer {i}",
            x=genome_w_positions[start_i:stop_i, 0],
            y=genome_w_positions[start_i:stop_i, 1],
            z=genome_w_positions[start_i:stop_i, 2],
            mode="markers",
            legendrank=1,
        )

    fig.update_layout(
        title=title.stem if isinstance(title, Path) else title.split("/")[-1],
        scene_camera={
            "up": dict(x=0, y=0, z=1),
            "center": dict(x=0, y=0, z=0),
            "eye": dict(x=-1.5, y=1.5, z=1.5),
        },
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # save as png and html
    fig.write_html(f"{title}.html")
    fig.write_image(f"{title}.png")
