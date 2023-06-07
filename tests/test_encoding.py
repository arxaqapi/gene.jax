import unittest
from math import prod

import jax.numpy as jnp

from gene.encoding import direct_decoding, direct_enc_genome_size, gene_enc_genome_size


class TestDirectEncoding(unittest.TestCase):
    def test_values_direct_encoding(self):
        layer_dims = [784, 64, 10]  # (784, 64) (64, 10)
        # (50816,)
        genome_w_length = prod(layer_dims[:2]) + prod(layer_dims[1:])
        genome_b_length = sum(layer_dims[1:])
        genome = jnp.arange(start=0, stop=genome_w_length + genome_b_length)

        model_param = direct_decoding(
            genome,
            config={"net": {"layer_dimensions": layer_dims}, "encoding": {"d": 1}},
        )

        self.assertEqual(model_param["Dense_0"]["kernel"][10][15], (10 * 64 + 15))
        self.assertEqual(
            model_param["Dense_1"]["kernel"][32][3], (784 * 64 + 32 * 10 + 3)
        )

        self.assertEqual(model_param["Dense_0"]["bias"][0], 50816)
        self.assertEqual(model_param["Dense_1"]["bias"][-1], genome[-1])

        self.assertEqual(model_param["Dense_0"]["kernel"].shape, (784, 64))
        self.assertEqual(model_param["Dense_0"]["bias"].shape, (64,))
        self.assertEqual(model_param["Dense_1"]["kernel"].shape, (64, 10))
        self.assertEqual(model_param["Dense_1"]["bias"].shape, (10,))

    def test_direct_enc_genome_size(self):
        """Test the function `direct_enc_genome_size`,
        so that the generated genome has the correct size.
        """
        layer_dims = [784, 64, 10]  # (784, 64) (64, 10)
        # (50816,)
        genome_w_length = prod(layer_dims[:2]) + prod(layer_dims[1:])
        genome_b_length = sum(layer_dims[1:])

        self.assertEqual(
            genome_w_length + genome_b_length,
            direct_enc_genome_size(
                {"net": {"layer_dimensions": layer_dims}, "encoding": {"d": 1}}
            ),
        )


class TestGenomeSize(unittest.TestCase):
    def test_genome_size_paper(self):
        layer_dimensions = [128, 64, 64, 18]
        d = 3
        n = sum(layer_dimensions)

        self.assertEqual(n * (d + 1), 1096)
        self.assertEqual(
            gene_enc_genome_size(
                {"encoding": {"d": d}, "net": {"layer_dimensions": layer_dimensions}}
            ),
            968,
        )
