import unittest

import jax
import jax.numpy as jnp
import numpy as np
from stafe import deepax


class TestDense(unittest.TestCase):
    def test_forward_dims(self):
        input_data = jnp.ones((128, 20))
        model = deepax.Linear(20, 30, rng=deepax.Generator())
        out = model(input_data)
        # model.W[0][0] == 0.002713859
        self.assertEqual(out.shape, (128, 30))
    
    def test_sequential_forward_dims(self):
        rng = deepax.Generator(0)
        input_data = jnp.ones((1000, 784))
        model = deepax.Sequential(
            deepax.Linear(784, 128, rng=rng),
            jax.nn.relu,
            deepax.Linear(128, 10, rng=rng))
        out = model(input_data)

        self.assertEqual(out.shape, (1000, 10))

    def test_sequential_custom_para_forward_dims(self):
        rng = deepax.Generator(0)
        input_data = jnp.ones((1000, 784))

        model = deepax.Sequential(
            deepax.Linear(784, 128, rng=rng, parameters= {
                'w': jnp.zeros((784, 128)),
                'b': jnp.zeros((128, ))
            }),
            deepax.ReLU(),
            deepax.Linear(128, 10, rng=rng, parameters= {
                'w': jnp.zeros((128, 10)),
                'b': jnp.zeros((10, ))
            }))
        out = model(input_data)
        
        self.assertEqual(model.layers[0].parameters['w'][38][8], 0.)
        self.assertEqual(out.shape, (1000, 10))
    

class TestFlatNet(unittest.TestCase):
    def test_flatnet_forward(self):
        model = deepax.FlatNet(
            genome=jax.random.uniform(
                jax.random.PRNGKey(2),
                ((128 + 64 + 18) * 4, ),
                minval=-10,
                maxval=10),
            dimensions=[128, 64, 18])
        
        synthetic_genomes = [
            jnp.full((128, ), fill_value=100),
            jnp.full((128, ), fill_value=-100),
            jax.random.uniform(jax.random.PRNGKey(1), (128, ))]

        for genome in synthetic_genomes:
            out = model.forward(genome)
            print(out)

    def test_flatnet__full_forward(self):
        model = deepax.FlatNet(
            genome=jax.random.uniform(
                jax.random.PRNGKey(2),
                ((128 + 64 + 64 + 18) * 4, ),
                minval=-10,
                maxval=10))
        
        synthetic_genomes = [
            jnp.full((128, ), fill_value=100),
            jnp.full((128, ), fill_value=-100),
            jax.random.uniform(jax.random.PRNGKey(1), (128, ))]

        for genome in synthetic_genomes:
            out = model.forward(genome)
            print(out)


class TestGenerator(unittest.TestCase):
    def test_generator_genkeys(self):
        gen = deepax.Generator()
        old_key = gen.base_key.copy()
        _ = gen.create_subkeys(1)

        self.assertIsNone(
            np.testing.assert_array_compare(lambda a, b : a != b, old_key, gen.base_key))

    def test_generator_subkeys(self):
        gen = deepax.Generator()
        subkey_1 = gen.create_subkeys(1)
        subkey_2 = gen.create_subkeys(1)

        self.assertIsNone(
            np.testing.assert_array_compare(lambda a, b : a != b, subkey_1, subkey_2))
        
    def test_generator_number_subkeys(self):
        gen = deepax.Generator()
        n = np.random.randint(10)
        subkeys = gen.create_subkeys(n)
        self.assertEqual(n, len(subkeys))