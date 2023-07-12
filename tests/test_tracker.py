import unittest

import jax.random as jrd
import jax.numpy as jnp
import chex

from gene.tracker import Tracker
from gene.core.decoding import GENEDecoder
from gene.core.distances import pL2Distance


class TestTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "evo": {
                "n_generations": 30,
                "population_size": 200,
            },
            "net": {
                "layer_dimensions": [18, 128, 128, 6],
                "architecture": "tanh_linear",
            },
            "encoding": {"type": "gene", "d": 3},
            "task": {
                "maximize": True,
            },
        }
        self.decoder = GENEDecoder(self.config, pL2Distance())

    def test_best_carry_over(self):
        """
        Test that the best individuals, and its fitness is correctly carried
        over generations
        """
        # Gen 0
        population_gen_0 = jnp.array(
            [
                [1.0] * self.decoder.encoding_size(),
                [3.0] * self.decoder.encoding_size(),
                [5.0] * self.decoder.encoding_size(),
                [7.0] * self.decoder.encoding_size(),
                [9.0] * self.decoder.encoding_size(),
                [11.0] * self.decoder.encoding_size(),
            ]
        )
        # arg order = (5, 2, 4)
        pop_gen_0_fit = jnp.array([2.0, -5.0, 6.0, 0.0, 3.0, 5555.0])

        tracker = Tracker(self.config, self.decoder)
        tracker_state = tracker.init()

        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=population_gen_0,
            fitnesses=pop_gen_0_fit,
            sample_mean=None,
            eval_f=(lambda *_: None),
            rng_eval=None,
        )

        # check ordering of the best individuals
        for i, eq_idx in enumerate([5, 2, 4]):
            self.assertIsNone(
                chex.assert_trees_all_equal(
                    tracker_state["backup"]["top_k_individuals"][i],
                    population_gen_0[eq_idx],
                )
            )
        # check ordering of the fitness
        for i, eq_idx in enumerate([5, 2, 4]):
            self.assertIsNone(
                chex.assert_trees_all_equal(
                    tracker_state["training"]["top_k_fit"][0][i],
                    pop_gen_0_fit[eq_idx],
                )
            )
        # Gen 1
        population_gen_1 = jnp.array(
            [
                [0.0] * self.decoder.encoding_size(),
                [2.0] * self.decoder.encoding_size(),
                [4.0] * self.decoder.encoding_size(),
                [6.0] * self.decoder.encoding_size(),
                [8.0] * self.decoder.encoding_size(),
                [10.0] * self.decoder.encoding_size(),
            ]
        )
        # arg order = (2, 3, 5)
        pop_gen_1_fit = jnp.array([-5.3, 1.0, 1624.0, 7.6, -3.0, 3.0])

        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=population_gen_1,
            fitnesses=pop_gen_1_fit,
            sample_mean=None,
            eval_f=(lambda *_: None),
            rng_eval=None,
        )

        # check ordering of the best individuals
        for i, indiv_value in enumerate(jnp.array([11.0, 4.0, 6.0])):
            self.assertIsNone(
                chex.assert_trees_all_equal(
                    tracker_state["backup"]["top_k_individuals"][i][0],
                    indiv_value,
                )
            )
        # check ordering of the fitness
        for i, value in enumerate(jnp.array([5555.0, 1624.0, 7.6])):
            self.assertIsNone(
                chex.assert_trees_all_equal(
                    tracker_state["training"]["top_k_fit"][1][i],
                    value,
                )
            )

    def test_training_top_k_fit(self):
        tracker = Tracker(self.config, self.decoder, 3)
        tracker_state = tracker.init()

        size = GENEDecoder(self.config, pL2Distance()).encoding_size()
        # Individuals =>
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            individuals=jnp.ones((6, size)),
            fitnesses=jnp.array([1, 2.0, 6.0, 8.0, 2.0, 3.0]),
            sample_mean=None,
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
            sample_mean=None,
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
        tracker = Tracker(self.config, self.decoder, 3)
        tracker_state = tracker.init()

        size = self.decoder.encoding_size()

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
            sample_mean=None,
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

    # def test_training_empirical_mean_fit(self):
    #     tracker = Tracker(self.config, self.decoder, 3)
    #     t_state = tracker.init()

    #     t_state = tracker.update(
    #         tracker_state=t_state,
    #         individuals=jnp.zeros((self.decoder.encoding_size(),)),
    #         fitnesses=jnp.array([0.0, 2.0, 4.0]),
    #         sample_mean=None,
    #         eval_f=(lambda *_: None),
    #         rng_eval=jrd.PRNGKey(0),
    #     )

    #     self.assertIsNone(
    #         chex.assert_trees_all_close(
    #             2.0, t_state["training"]["empirical_mean_fit"][0]
    #         )
    #     )
    #     t_state = tracker.update(
    #         tracker_state=t_state,
    #         individuals=jnp.zeros((self.decoder.encoding_size(),)),
    #         fitnesses=jnp.array([10.0, 22.0, -23.0]),
    #         sample_mean=None,
    #         eval_f=(lambda *_: None),
    #         rng_eval=jrd.PRNGKey(0),
    #     )

    #     self.assertIsNone(
    #         chex.assert_trees_all_close(
    #             2.0, t_state["training"]["empirical_mean_fit"][0]
    #         )
    #     )
    #     self.assertIsNone(
    #         chex.assert_trees_all_close(
    #             3.0, t_state["training"]["empirical_mean_fit"][1]
    #         )
    #     )
