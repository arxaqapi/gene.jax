import unittest

import jax.random as jrd

from gene.core.models import ReluLinearModel
from gene.nn_properties import (
    # expressivity_ratio,
    # initialization_term,
    input_distribution_restoration,
)


class TestNNProps(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_input_distribution_restoration(self):
        rng = jrd.PRNGKey(0)
        rng_data, rng_init = jrd.split(rng, 2)

        x = jrd.normal(rng_data, (10,))
        model = ReluLinearModel([32, 32, 2])
        model_parameters = model.init(rng_init, x)

        (input_mu, input_sigma), (
            output_mu,
            output_sigma,
        ) = input_distribution_restoration(model, model_parameters, batch_size=100)
        print(f"{input_mu=} | {input_sigma=}")
        print(f"{output_mu=} | {output_sigma=}")

        self.assertIsNotNone(input_mu)
        self.assertIsNotNone(input_sigma)
        self.assertIsNotNone(output_mu)
        self.assertIsNotNone(output_sigma)
