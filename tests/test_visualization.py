import unittest

import jax.numpy as jnp

from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.core.models import BoundedLinearModelConf
from gene.core.distances import Distance_functions
from gene.core.decoding import GENEDecoder


class TestVizBrax(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
            "net": {"layer_dimensions": [18, 128, 128, 6]},
            "task": {"environnment": "halfcheetah", "episode_length": 1000},
        }
        self.model = BoundedLinearModelConf(self.config)
        self.df = Distance_functions[self.config["encoding"]["distance"]]()
        self.genome = jnp.zeros((GENEDecoder(self.config, self.df).encoding_size(),))

    def test_output_file_v1(self):
        self.assertIsNone(
            render_brax(
                *visualize_brax(self.config, self.genome, self.model, self.df),
                output_file="test_file",
            )
        )

    def test_output_file_v2(self):
        self.assertIsNone(
            render_brax(
                *visualize_brax(self.config, self.genome, self.model, self.df),
                output_file="test_file_2",
            )
        )
