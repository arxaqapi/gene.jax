import unittest

import jax.random as jrd
import jax.numpy as jnp
import chex

from gene.v1.tracker import Tracker
from gene.core.decoding import GENEDecoder
from gene.core.distances import pL2Distance


class TestTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "evo": {"n_generations": 30},
            "net": {"layer_dimensions": [17, 128, 128, 6]},
            "encoding": {"type": "gene", "d": 3},
            "task": {"maximize": True},
        }

    def test_training_top_k_fit(self):
        tracker = Tracker(self.config, 3)
        tracker_state = tracker.init()

        size = GENEDecoder(self.config, pL2Distance()).encoding_size()
        # Individuals =>
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=jnp.ones((6, size)),
            fitnesses=jnp.array([1, 2.0, 6.0, 8.0, 2.0, 3.0]),
            mean_ind=None,
            eval_f=(lambda *_: None),
            rng_eval=jrd.PRNGKey(0),
        )

        self.assertIsNone(
            chex.assert_trees_all_close(
                jnp.array([8.0, 6.0, 3.0]), tracker_state["training"]["top_k_fit"][0]
            )
        )

        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=jnp.ones((6, size)),
            fitnesses=jnp.array([13.0, -6.0, 4.0]),
            mean_ind=None,
            eval_f=(lambda *_: None),
            rng_eval=jrd.PRNGKey(0),
        )

        self.assertIsNone(
            chex.assert_trees_all_close(
                jnp.array([13.0, 8.0, 6.0]),
                tracker_state["training"]["top_k_fit"][1],
            )
        )

    def test_backup_top_k(self):
        # Test if the save top 3 individuals  are the correct ones and are in order
        tracker = Tracker(self.config, 3)
        tracker_state = tracker.init()

        size = GENEDecoder(self.config, pL2Distance()).encoding_size()

        base_shape = tracker_state["backup"]["top_k_individuals"].shape

        individuals = jnp.array(
            [
                jnp.empty((size,)),
                jnp.empty((size,)),
                jnp.full((size,), fill_value=1.0),
                jnp.full((size,), fill_value=0.0),
                jnp.empty((size,)),
                jnp.full((size,), fill_value=2.0),
            ]
        )

        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=individuals,
            fitnesses=jnp.array([1, 2.0, 6.0, 8.0, 2.0, 3.0]),  # args[3, 2, 5]
            mean_ind=None,
            eval_f=(lambda *_: None),
            rng_eval=jrd.PRNGKey(0),
        )

        # Check that shapes are the same
        self.assertEqual(base_shape, tracker_state["backup"]["top_k_individuals"].shape)

        # check top 3 are the correct ones
        for i, ind in enumerate(tracker_state["backup"]["top_k_individuals"]):
            self.assertIsNone(
                chex.assert_trees_all_close(ind, jnp.full((size,), fill_value=i))
            )

    def test_training_empirical_mean_fit(self):
        tracker = Tracker(self.config, 3)
        t_state = tracker.init()

        t_state = tracker.update(
            tracker_state=t_state,
            individuals=jnp.zeros((3,)),
            fitnesses=jnp.array([0.0, 2.0, 4.0]),
            mean_ind=None,
            eval_f=(lambda *_: None),
            rng_eval=jrd.PRNGKey(0),
        )

        self.assertIsNone(
            chex.assert_trees_all_close(
                2.0, t_state["training"]["empirical_mean_fit"][0]
            )
        )
        t_state = tracker.update(
            tracker_state=t_state,
            individuals=jnp.zeros((3,)),
            fitnesses=jnp.array([10.0, 22.0, -23.0]),
            mean_ind=None,
            eval_f=(lambda *_: None),
            rng_eval=jrd.PRNGKey(0),
        )

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
