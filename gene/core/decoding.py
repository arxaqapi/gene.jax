from __future__ import annotations

from jax import Array, lax
import jax.numpy as jnp
import flax.linen as nn


from gene.core.distances import Distance_functions


Phenotype = nn.FrozenDict


class Decoder:
    """A `Decoder` is a class that transforms a genotype into a phenotype."""

    def __init__(self, config: dict, *args, **kwargs) -> None:
        self.config = config

    def decode(self, genotype: Array) -> Phenotype:
        """Takes in a genome defined as the `genotype`, and outputs its corresponding `Phenotype`.

        Args:
            genotype (Array): The genome we try to decode, using the stored
                distance function.

        Returns:
            Phenotype: The corresponding `Phenotype`, as a `nn.FrozenDict` structure.
        """
        raise NotImplementedError

    def encoding_size(self) -> int:
        """Returns the size of the `genotype` based on the current run configuration.

        Returns:
            int: size of the genotype
        """
        raise NotImplementedError


# TODO - Test
class DirectDecoder(Decoder):
    """The `DirectDecoder` is a class that transforms a genotype into a phenotype,
    using direct encoding. Where each parameter is a single gene.
    """
    def __init__(self, config: dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

    def encoding_size(self) -> int:
        return sum(
            in_f * out_f + out_f
            for in_f, out_f in zip(
                self.config["net"]["layer_dimensions"][:-1],
                self.config["net"]["layer_dimensions"][1:],
            )
        )

    def decode(self, genotype: Array) -> Phenotype:
        """Each weight and bias of the neural network is encoded as a single gene
        in the genome.

        Config-less function.
        """
        layer_dims = self.config["net"]["layer_dimensions"]

        genome_w, genome_b = jnp.split(
            genotype,
            [
                sum(
                    layer_dims[i] * layer_dims[i + 1]
                    for i in range(len(layer_dims) - 1)
                )
            ],
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

        return nn.FrozenDict({"params": model_parameters})


# TODO - test me
class GENEDecoder(Decoder):
    """The `GENEDecoder`, is a class that transforms a genotype into a phenotype,
    using a specific distance function, saved as `distance_function`,
    and optional parameters.
    """
    def __init__(self, config: dict, distance_function: "Distance.DistanceFunction", *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.distance_function = distance_function
        # self.distance_function = Distance_functions[config["encoding"]["distance"]]

    def encoding_size(self) -> int:
        """Size of the awaited genotype GENE encoding

        Returns:
            int: _description_
        """
        return self.config["net"]["layer_dimensions"][0] * self.config["encoding"][
            "d"
        ] + sum(self.config["net"]["layer_dimensions"][1:]) * (
            self.config["encoding"]["d"] + 1
        )

    def decode(
        self,
        genotype: Array,
    ):
        layer_dims = self.config["net"]["layer_dimensions"]
        d = self.config["encoding"]["d"]

        # To facilitate acces to the encoding of the weights and the biases
        # (and reduce confusion and possible error in computing indexes),
        # we split the genome in 2 parts
        genome_w, genome_b = jnp.split(genotype, [sum(layer_dims) * d])

        model_parameters: nn.FrozenDict = {}
        for i, (layer_in, layer_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            # Split genome into subarrays, where each is the position vector for one neuron
            genome_w_positions = jnp.array(jnp.split(genome_w, sum(layer_dims)))

            layer_offset = sum(layer_dims[:i])
            # indexes of the previous layer neurons
            src_idx = layer_offset + jnp.arange(start=0, stop=layer_in)
            # indexes of the current layer neurons
            target_idx = layer_offset + layer_in + jnp.arange(start=0, stop=layer_out)

            weight_matrix = self.distance_function.vectorized_measure(
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
        return nn.FrozenDict({"params": model_parameters})


Decoders = {
    "direct": DirectDecoder,
    "gene": GENEDecoder,
}