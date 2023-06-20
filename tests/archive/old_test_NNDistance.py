import unittest
from pathlib import Path

import jax.numpy as jnp
import chex

from gene.learn_distance import NNDistance
from gene.encoding import direct_enc_genome_size


class TestNNDistance(unittest.TestCase):
    def test_save(self):
        # init config
        config = {"net": {"layer_dimensions": [4, 32, 32, 1]}}
        genome = jnp.ones((direct_enc_genome_size(config),))
        # init dist
        distance = NNDistance(
            distance_genome=genome,
            layer_dimensions=config["net"]["layer_dimensions"],
        )

        save_f = Path("test.aze")
        distance.save_parameters(save_f)

        pytree = distance.model_parameters

        self.assertTrue(save_f.with_suffix(".pkl").exists())

        distance.load_parameters(save_f)
        pytree_unpickled = distance.model_parameters

        self.assertIsNone(chex.assert_trees_all_equal(pytree, pytree_unpickled))

        # remove artifacts
        save_f.with_suffix(".pkl").unlink()
