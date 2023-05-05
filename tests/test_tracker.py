import unittest
import chex
import jax
import jax.numpy as jnp

from gene.tracker import Tracker


class TestTrackerObject(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "evo": {
                "n_generations": 30,
            },
        }

    def test_training_top_k_fit(self):
        tracker = Tracker(self.config, 3)
        t_state = tracker.init()

        t_state = tracker.update(t_state, None, jnp.array([1, 2.0, 6.0, 8.0, 2.0, 3.0]))
        self.assertIsNone(
            chex.assert_trees_all_close(
                jnp.array([8.0, 6.0, 3.0]), t_state["training"]["top_k_fit"][0]
            )
        )

        t_state = tracker.update(t_state, None, jnp.array([13.0, -6.0, 4.0]))
        self.assertIsNone(
            chex.assert_trees_all_close(
                jnp.array([13.0, 8.0, 6.0]),
                t_state["training"]["top_k_fit"][1],
            )
        )

    def test_training_empirical_mean_fit(self):
        tracker = Tracker(self.config, 3)
        t_state = tracker.init()

        t_state = tracker.update(t_state, None, jnp.array([0.0, 2.0, 4.0]))

        self.assertIsNone(
            chex.assert_trees_all_close(
                2.0, t_state["training"]["empirical_mean_fit"][0]
            )
        )
        t_state = tracker.update(t_state, None, jnp.array([10.0, 22.0, -23.0]))
        self.assertIsNone(
            chex.assert_trees_all_close(
                2.0, t_state["training"]["empirical_mean_fit"][0]
            )
        )
        self.assertIsNone(
            chex.assert_trees_all_close(
                3.0, t_state["training"]["empirical_mean_fit"][1]
            )
        )
