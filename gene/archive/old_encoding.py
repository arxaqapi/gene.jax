import jax.numpy as jnp
from jax import lax
import flax.linen as nn

from gene.network import BoundedLinearModel
from gene.distances import Vectorized_distances


def _gene_enc_genome_size(layer_dims: tuple[int], d: int) -> int:
    return layer_dims[0] * d + sum(layer_dims[1:]) * (d + 1)


def gene_enc_genome_size(config: dict):
    """Computes the effective size of the genome based on the
    layers dimensionnalities.
    """
    # The first value in layer_dimension does is only used for the dimensionnality
    # of the input features. So biases are attributed to it
    return _gene_enc_genome_size(
        config["net"]["layer_dimensions"], config["encoding"]["d"]
    )


def _direct_enc_genome_size(layer_dims: list[int]) -> int:
    return sum(
        in_f * out_f + out_f for in_f, out_f in zip(layer_dims[:-1], layer_dims[1:])
    )


def direct_enc_genome_size(config: dict):
    layer_dims = config["net"]["layer_dimensions"]

    return _direct_enc_genome_size(layer_dims)


# Decoding functions


def gene_decoding(genome: jnp.ndarray, config: dict):
    assert genome.shape[0] == gene_enc_genome_size(config)
    layer_dims = config["net"]["layer_dimensions"]
    d = config["encoding"]["d"]

    # To facilitate acces to the encoding of the weights and the biases
    # (and reduce confusion and possible error in computing indexes),
    # we split the genome in 2 parts
    genome_w, genome_b = jnp.split(genome, [sum(layer_dims) * d])

    model_parameters: nn.FrozenDict = {}
    for i, (layer_in, layer_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        # Split genome into subarrays, where each is the position vector for one neuron
        genome_w_positions = jnp.array(jnp.split(genome_w, sum(layer_dims)))

        layer_offset = sum(layer_dims[:i])
        # indexes of the previous layer neurons
        src_idx = layer_offset + jnp.arange(start=0, stop=layer_in)
        # indexes of the current layer neurons
        target_idx = layer_offset + layer_in + jnp.arange(start=0, stop=layer_out)

        weight_matrix = Vectorized_distances[config["encoding"]["distance"]](
            genome_w_positions, src_idx, target_idx
        )
        # Biases are directly encoded into the genome,
        # they are stored at the end of the genome, in genome_b
        biases = lax.dynamic_slice(
            genome_b, (sum(layer_dims[1 : i + 1]),), (layer_out,)
        )

        model_parameters[f"Dense_{i}"] = {
            "kernel": weight_matrix,
            "bias": biases,
        }
    # FIXME - return frozen dict model params directly
    return model_parameters


def gene_decoding_w_dist(genome: jnp.ndarray, config: dict, distance_network) -> dict:
    """Uses the `distance_network` to compute the distance between two neurons.

    Args:
        genome (jnp.ndarray): The gene encoded genome to decode
        config (dict): config dict containig the current run informations
        distance_network (NNDistance): parametrized distance function.

    Returns:
        dict: decoded model parameters
    """
    assert genome.shape[0] == gene_enc_genome_size(config)
    layer_dims = config["net"]["layer_dimensions"]
    d = config["encoding"]["d"]

    genome_w, genome_b = jnp.split(genome, [sum(layer_dims) * d])

    model_parameters: nn.FrozenDict = {}
    for i, (layer_in, layer_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        genome_w_positions = jnp.array(jnp.split(genome_w, sum(layer_dims)))

        layer_offset = sum(layer_dims[:i])
        src_idx = layer_offset + jnp.arange(start=0, stop=layer_in)
        target_idx = layer_offset + layer_in + jnp.arange(start=0, stop=layer_out)

        # NOTE - Use the network in distance_network to compute the distance
        # NOTE - Squeeze necessary since
        # params = {'Dense_0': {'bias': (128,), 'kernel': (17, 128, 1)},
        #           'Dense_1': {'bias': (128,), 'kernel': (128, 128, 1)},
        #           'Dense_2': {'bias': (6,),   'kernel': (128, 6, 1)}}
        weight_matrix = distance_network.vmap_evaluate(
            genome_w_positions, src_idx, target_idx
        )
        weight_matrix = jnp.squeeze(weight_matrix)
        biases = lax.dynamic_slice(
            genome_b, (sum(layer_dims[1 : i + 1]),), (layer_out,)
        )

        model_parameters[f"Dense_{i}"] = {
            "kernel": weight_matrix,
            "bias": biases,
        }

    return model_parameters


def _direct_decoding(genome: jnp.ndarray, layer_dims: list[int]):
    """Each weight and bias of the neural network is encoded as a single gene
    in the genome.

    Config-less function.
    """
    genome_w, genome_b = jnp.split(
        genome,
        [sum(layer_dims[i] * layer_dims[i + 1] for i in range(len(layer_dims) - 1))],
    )

    model_parameters: dict = {}
    offset = 0
    for i, (layer_in, layer_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        section_length = layer_in * layer_out
        weights = lax.dynamic_slice(genome_w, (offset,), (section_length,))

        weight_matrix = jnp.reshape(weights, newshape=(layer_in, layer_out))
        biases = lax.dynamic_slice(
            genome_b, (sum(layer_dims[1 : i + 1]),), (layer_out,)
        )

        model_parameters[f"Dense_{i}"] = {
            "kernel": weight_matrix,
            "bias": biases,
        }

        offset += section_length

    return model_parameters


def direct_decoding(genome: jnp.ndarray, config: dict):
    """Each weight and bias of the neural network is encoded as a single gene
    in the genome.
    """
    layer_dims = config["net"]["layer_dimensions"]

    return _direct_decoding(genome, layer_dims)


def genome_to_model(
    genome: jnp.ndarray, config: dict
) -> tuple[BoundedLinearModel, nn.FrozenDict]:
    model_parameters = Encoding_function[config["encoding"]["type"]](genome, config)

    model = BoundedLinearModel(config["net"]["layer_dimensions"][1:])

    return model, nn.FrozenDict({"params": model_parameters})


Encoding_size_function = {
    "direct": direct_enc_genome_size,
    "gene": gene_enc_genome_size,
}
Encoding_function = {"direct": direct_decoding, "gene": gene_decoding}
