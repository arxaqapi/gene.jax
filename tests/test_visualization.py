import unittest
import os

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
            "task": {"environnment": "halfcheetah", "episode_length": 100},
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
        # https://github.com/google/jax/issues/8916
        # works on CPU but not GPU, try it on bigger GPU
        self.config["net"]["layer_dimensions"][0] = 17
        self.assertIsNone(
            render_brax(
                *visualize_brax(
                    self.config, self.genome, self.model, self.df, use_v1=False
                ),
                output_file="test_file_2",
            )
        )

    def tearDown(self) -> None:
        if os.path.exists("test_file.html"):
            os.remove("test_file.html")
        if os.path.exists("test_file_2.html"):
            os.remove("test_file_2.html")
