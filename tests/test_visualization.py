import unittest
import os

import jax.numpy as jnp
import jax.random as jrd

from gene.visualize.visualize_brax import visualize_brax, render_brax
from gene.visualize.neurons import visualize_neurons_2d, visualize_neurons_3d
from gene.core.models import ReluTanhLinearModelConf
from gene.core.distances import Distance_functions, pL2Distance
from gene.core.decoding import GENEDecoder


class TestVizBrax(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
            "net": {"layer_dimensions": [18, 128, 128, 6]},
            "task": {"environnment": "halfcheetah", "episode_length": 100},
        }
        self.model = ReluTanhLinearModelConf(self.config)
        self.df = Distance_functions[self.config["encoding"]["distance"]]()
        self.genome = jnp.zeros((GENEDecoder(self.config, self.df).encoding_size(),))

    def test_output_file_v1(self):
        self.assertIsNone(
            render_brax(
                "test_file",
                *visualize_brax(self.config, self.genome, self.model, self.df),
            )
        )

    def test_output_file_v2(self):
        # https://github.com/google/jax/issues/8916
        # works on CPU but not GPU, try it on bigger GPU
        self.config["net"]["layer_dimensions"][0] = 17
        self.assertIsNone(
            render_brax(
                "test_file_2",
                *visualize_brax(
                    self.config, self.genome, self.model, self.df, use_v1=False
                ),
            )
        )

    def tearDown(self) -> None:
        if os.path.exists("test_file.html"):
            os.remove("test_file.html")
        if os.path.exists("test_file_2.html"):
            os.remove("test_file_2.html")


class TestVisualizeNeurons(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "net": {"layer_dimensions": [10, 32, 32, 1]},
            "encoding": {"d": 2},
        }

    def test_visualize_neurons_2d(self):
        genome_size = GENEDecoder(self.config, pL2Distance()).encoding_size()
        genome = jrd.normal(jrd.PRNGKey(0), shape=(genome_size,))
        self.assertIsNone(
            visualize_neurons_2d(genome, self.config, title="test_neuron_viz_2d")
        )

    def test_visualize_neurons_3d(self):
        self.config["encoding"]["d"] = 3

        genome_size = GENEDecoder(self.config, pL2Distance()).encoding_size()
        genome = jrd.normal(jrd.PRNGKey(0), shape=(genome_size,))

        self.assertIsNone(
            visualize_neurons_3d(genome, self.config, title="test_neuron_viz_3d")
        )

    def tearDown(self) -> None:
        for f in ["test_neuron_viz_2d", "test_neuron_viz_3d"]:
            for f_extension in ["html", "png"]:
                f_path = f"{f}.{f_extension}"
                if os.path.exists(f_path):
                    os.remove(f_path)
