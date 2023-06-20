import unittest
from math import prod

import jax.numpy as jnp

from gene.encoding import direct_decoding, direct_enc_genome_size, gene_enc_genome_size


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
