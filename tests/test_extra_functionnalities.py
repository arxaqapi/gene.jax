import unittest
import numpy as np
import jax.numpy as jnp
import jax


class TestRandomSort(unittest.TestCase):
    def test_sort_argsort(self):
        r_vec = jax.random.randint(key=jax.random.PRNGKey(0), shape=(10,),minval=0, maxval=10)

        self.assertIsNone(
            np.testing.assert_array_equal(
                jnp.sort(r_vec),
                r_vec[jnp.argsort(r_vec)]    
            )
        )
    def test_sort_argsort_decreasing_order(self):
        r_vec = jax.random.randint(key=jax.random.PRNGKey(0), shape=(10,),minval=0, maxval=10)
        max_val = r_vec.max()

        self.assertEqual(
            max_val,
            r_vec[jnp.argsort(r_vec)[::-1]][0]
        )