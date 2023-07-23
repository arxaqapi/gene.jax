import unittest

import jax.numpy as jnp
import jax.random as jrd
from jax import Array

from gene.learning import (
    brax_eval_n_times,
    learn_gymnax_task,
    learn_brax_task_untracked,
)
from gene.core.decoding import GENEDecoder, DirectDecoder
from gene.core.distances import pL2Distance
from gene.core.evaluation import get_braxv1_env
from gene.core.models import ReluLinearModel
from gene.utils import load_config


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


class TestMetaReadyLearningFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("config/nn_meta_df.json")
        self.meta_decoder = DirectDecoder(self.config)
        self.nn_dst_model = ReluLinearModel(self.config["net"]["layer_dimensions"][1:])

    def test_learn_gymnax_task(self):
        rng = jrd.PRNGKey(0)

        out = learn_gymnax_task(
            df_genotype=jnp.ones((self.meta_decoder.encoding_size(),)),
            rng=rng,
            meta_decoder=self.meta_decoder,
            df_model=self.nn_dst_model,
            config=self.config["curriculum"]["cart"],
        )

        self.assertIsNotNone(out)

    def test_learn_brax_task_untracked(self):
        rng = jrd.PRNGKey(0)

        out = learn_brax_task_untracked(
            df_genotype=jnp.ones((self.meta_decoder.encoding_size(),)),
            rng=rng,
            meta_decoder=self.meta_decoder,
            df_model=self.nn_dst_model,
            config=self.config["curriculum"]["hc_100"],
        )

        self.assertIsNotNone(out)
