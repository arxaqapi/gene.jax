import unittest
from math import prod

import jax.numpy as jnp

from gene.core import decoding, distances


class TestBaseDecoder(unittest.TestCase):
    def test_init(self):
        config = {}
        d = decoding.Decoder(config, None, None)

        self.assertTrue(type(d) is decoding.Decoder)


class TestDirectDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"net": {"layer_dimensions": [784, 64, 10]}, "encoding": {"d": 1}}
        self.decoder = decoding.DirectDecoder(self.config)

    def test_values_direct_encoding(self):
        layer_dims = self.config["net"]["layer_dimensions"]

        genome_w_length = prod(layer_dims[:2]) + prod(layer_dims[1:])
        genome_b_length = sum(layer_dims[1:])
        genome = jnp.arange(start=0, stop=genome_w_length + genome_b_length)

        model_param = self.decoder.decode(genome)

        self.assertEqual(
            model_param["params"]["Dense_0"]["kernel"][10][15], (10 * 64 + 15)
        )
        self.assertEqual(
            model_param["params"]["Dense_1"]["kernel"][32][3], (784 * 64 + 32 * 10 + 3)
        )

        self.assertEqual(model_param["params"]["Dense_0"]["bias"][0], 50816)
        self.assertEqual(model_param["params"]["Dense_1"]["bias"][-1], genome[-1])

        self.assertEqual(model_param["params"]["Dense_0"]["kernel"].shape, (784, 64))
        self.assertEqual(model_param["params"]["Dense_0"]["bias"].shape, (64,))
        self.assertEqual(model_param["params"]["Dense_1"]["kernel"].shape, (64, 10))
        self.assertEqual(model_param["params"]["Dense_1"]["bias"].shape, (10,))

    def test_direct_enc_genome_size(self):
        """Test the function `direct_enc_genome_size`,
        so that the generated genome has the correct size.
        """
        layer_dims = [784, 64, 10]  # (784, 64) (64, 10)
        # (50816,)
        genome_w_length = prod(layer_dims[:2]) + prod(layer_dims[1:])
        genome_b_length = sum(layer_dims[1:])

        self.assertEqual(
            genome_w_length + genome_b_length, self.decoder.encoding_size()
        )


class TestGENEDecoder(unittest.TestCase):
    def test_genome_size_paper(self):
        layer_dimensions = [128, 64, 64, 18]
        d = 3
        n = sum(layer_dimensions)

        config = {"encoding": {"d": d}, "net": {"layer_dimensions": layer_dimensions}}
        decoder = decoding.GENEDecoder(config, distances.pL2Distance())

        self.assertEqual(n * (d + 1), 1096)
        self.assertEqual(
            decoder.encoding_size(),
            968,
        )
