from jax import jit
import jax.numpy as jnp



@jit
def test_split(x):
    return jnp.split(x, 1000)



if __name__ == "__main__":
    n = 1000
    d = 3
    genome = jnp.arange(0, n * d)
    result = test_split(genome)

    print(len(result))
    print(result[0].shape)
    print(result[:10])

