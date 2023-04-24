import jax.numpy as jnp
from jax import lax
import flax.linen as nn

from gene.network import LinearModel
from gene.utils import genome_size

from gene.distances import jit_vmap_distance_f


# TODO: remove below and wrap into _genome_to_model_func and partially apply evaluate_individual
# L2 dist vmap over 2 axis (returns matrix) and jit
_jit_vmap_L2_dist = jit_vmap_distance_f("L2")
# tag dist vmap over 2 axis (returns matrix) and jit
_jit_vmap_tag_dist = jit_vmap_distance_f("tag")


def gene_decoding(_genome: list[float], settings: dict):
    assert _genome.shape[0] == genome_size(settings)
    layer_dims = settings["net"]["layer_dimensions"]
    d = settings["encoding"]["d"]

    split_i = sum(layer_dims) * d

    # To facilitate acces to the encoding of the weights and the biases (and reduce confusion and possible error in computing indexes), we split the genome in 2 parts
    _genome_w, _genome_b = jnp.split(_genome, [split_i])

    model_parameters = {}
    for i, (layer_in, layer_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        # Split the genome into subarrays, each subarray is the position vector for one neuron
        genome_w_positions = jnp.array(jnp.split(_genome_w, sum(layer_dims)))

        layer_offset = sum(layer_dims[:i])
        # indexes of the previous layer neurons
        src_idx = layer_offset + jnp.arange(start=0, stop=layer_in)
        # indexes of the current layer neurons
        target_idx = layer_offset + layer_in + jnp.arange(start=0, stop=layer_out)

        weight_matrix = _jit_vmap_L2_dist(genome_w_positions, src_idx, target_idx)
        # Biases are directly encoded into the genome, they are stored at the end of the genome, in _genome_b
        biases = lax.dynamic_slice(
            _genome_b, (sum(layer_dims[1 : i + 1]),), (layer_out,)
        )

        model_parameters[f"Dense_{i}"] = {
            "kernel": weight_matrix,
            "bias": biases,
        }

    return model_parameters


def direct_decoding():
    """Each weight and bias of the neural network is encoded as a single gene in the genome"""
    raise NotImplementedError


def genome_to_model(genome: list[float], settings: dict):
    model_parameters = gene_decoding(genome, settings)
    model = LinearModel(settings["net"]["layer_dimensions"][1:])

    return model, nn.FrozenDict({"params": model_parameters})
