import unittest

# import jax.numpy as jnp
import jax.random as jrd
from jax import tree_util
import chex

from gene.core.models import (
    LinearModelConf,
    LinearModel,
    BoundedLinearModel,
    BoundedLinearModelConf,
    TanhLinearModel,
    TanhLinearModelConf,
    get_model,
)


class TestModelInit(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"net": {"layer_dimensions": [784, 64, 64, 10]}}
        self.BATCH_SIZE = 32

        # random keys
        rng = jrd.PRNGKey(0)
        self.rng_init, rng_data = jrd.split(rng, 2)
        # dummy data
        self.x_dummy = jrd.normal(rng_data, (self.BATCH_SIZE, 784))

    def test_compatibility_LinearModel(self):
        # model & model_param init & forward pass
        model_c = LinearModelConf(self.config)
        model_c_parameters = model_c.init(self.rng_init, self.x_dummy)
        out_c = model_c.apply(model_c_parameters, self.x_dummy)

        # normal linear model
        model = LinearModel(self.config["net"]["layer_dimensions"][1:])
        model_parameters = model.init(self.rng_init, self.x_dummy)
        out = model_c.apply(model_c_parameters, self.x_dummy)

        # print(tree_util.tree_map(lambda e: e.shape, model_c_parameters))

        # check that all leaves hae same shape and type, independent from
        # their classes
        self.assertIsNone(
            chex.assert_trees_all_equal_shapes_and_dtypes(
                tree_util.tree_leaves(model_c_parameters),
                tree_util.tree_leaves(model_parameters),
            )
        )
        # Check everything is the same, even the names
        self.assertIsNone(
            chex.assert_trees_all_close(model_c_parameters, model_parameters)
        )
        # Check output is the same
        self.assertIsNone(chex.assert_trees_all_close(out_c, out))

    def test_compatibility_BoundedLinearModel(self):
        # model & model_param init & forward pass
        model_c = BoundedLinearModelConf(self.config)
        model_c_parameters = model_c.init(self.rng_init, self.x_dummy)
        out_c = model_c.apply(model_c_parameters, self.x_dummy)

        # normal linear model
        model = BoundedLinearModel(self.config["net"]["layer_dimensions"][1:])
        model_parameters = model.init(self.rng_init, self.x_dummy)
        out = model_c.apply(model_c_parameters, self.x_dummy)

        # check that all leaves hae same shape and type, independent from
        # their classes
        self.assertIsNone(
            chex.assert_trees_all_equal_shapes_and_dtypes(
                tree_util.tree_leaves(model_c_parameters),
                tree_util.tree_leaves(model_parameters),
            )
        )
        # Check everything is the same, even the names
        self.assertIsNone(
            chex.assert_trees_all_close(model_c_parameters, model_parameters)
        )
        # Check output is the same
        self.assertIsNone(chex.assert_trees_all_close(out_c, out))

    def test_compatibility_TanhLinearModel(self):
        # model & model_param init & forward pass
        model_c = TanhLinearModelConf(self.config)
        model_c_parameters = model_c.init(self.rng_init, self.x_dummy)
        out_c = model_c.apply(model_c_parameters, self.x_dummy)

        # normal linear model
        model = TanhLinearModel(self.config["net"]["layer_dimensions"][1:])
        model_parameters = model.init(self.rng_init, self.x_dummy)
        out = model_c.apply(model_c_parameters, self.x_dummy)

        # check that all leaves hae same shape and type, independent from
        # their classes
        self.assertIsNone(
            chex.assert_trees_all_equal_shapes_and_dtypes(
                tree_util.tree_leaves(model_c_parameters),
                tree_util.tree_leaves(model_parameters),
            )
        )
        # Check everything is the same, even the names
        self.assertIsNone(
            chex.assert_trees_all_close(model_c_parameters, model_parameters)
        )
        # Check output is the same
        self.assertIsNone(chex.assert_trees_all_close(out_c, out))


class TestModelUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"net": {}}

    def test_get_model_base(self):
        self.assertTrue(type(get_model(self.config)) is BoundedLinearModelConf)

    def test_get_model_parametrized(self):
        # tests that get_model works as expected
        self.config["net"]["architecture"] = "linear"
        self.assertTrue(type(get_model(self.config)) is LinearModelConf)

        self.config["net"]["architecture"] = "bounded_linear"
        self.assertTrue(type(get_model(self.config)) is BoundedLinearModelConf)

        self.config["net"]["architecture"] = "tanh_linear"
        self.assertTrue(type(get_model(self.config)) is TanhLinearModelConf)
