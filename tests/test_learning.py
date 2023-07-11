import unittest

import jax.numpy as jnp
import jax.random as jrd
from jax import Array

from gene.learning import brax_eval_n_times
from gene.core.decoding import GENEDecoder
from gene.core.distances import pL2Distance
from gene.core.evaluation import get_braxv1_env


class TestLearningEvalLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "net": {
                "layer_dimensions": [18, 128, 128, 6],
                "architecture": "tanh_linear",
            },
            "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
            "task": {
                "environnment": "halfcheetah",
                "maximize": True,
                "episode_length": 1000,
            },
        }
        self.decoder = GENEDecoder(self.config, pL2Distance())

    def test_brax_eval_n_times(self):
        genome = jnp.ones((self.decoder.encoding_size(),))
        fitness = brax_eval_n_times(
            genome=genome,
            rng=jrd.PRNGKey(0),
            decoder=self.decoder,
            config=self.config,
            env=get_braxv1_env(self.config),
        )

        self.assertTrue(isinstance(fitness, Array))
