from numpy.distutils.misc_util import is_sequence
import jax.numpy as jnp
import jax 


def L2_gene(n1: jnp.ndarray, n2: jnp.ndarray) -> float:
    assert n1.shape[0] == n2.shape[0]
    """Compute the L2 distance between two points (neurons) represented as vectors

    Args:
        n1 (Array): Vector of values representing neuron n°1
        n2 (Array): Vector of values representing neuron n°2
        d (int, optional): Dimension of the coordinates. Defaults to 1.

    Returns:
        float: Measured distance between n1 and n2
    """
    # # diff = n1 - n2
    # # out = jnp.sqrt(diff.dot(diff)) # more efficient, square + sum together
    out = jnp.sqrt(jnp.sum(jnp.square(n1 - n2)))
    return out.astype(float)  # if is_sequence(out) else out


@jax.jit
def a_v(x: jnp.ndarray) -> jnp.ndarray:
    """Vectorrized and JIT-ed form of a()"""
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes
    x = jnp.where(x >= 1., x , 1.)
    x = jnp.where(x <= -1., x , -1.)
    return x


def a(x):
    """Identity function bounded to -1 and 1

    Args:
        x (float): Input value

    Returns:
        float | jnp.ndarray: Output value
    """
    if is_sequence(x): #type(x) in [Array, jnp.ndarray]:
        x[x >= 1] = 1
        x[x <= -1] = -1
    elif x >= 1:
        x = 1.
    elif x <= -1:
        x = -1.
    return x


def pL2_gene(n1: jnp.ndarray, n2: jnp.ndarray, d: int = 1) -> float:
    assert n1.shape[0] == n2.shape[0]
    assert n1.shape[0] == d
    return a_v(jnp.prod(n1 - n2)) * L2_gene(n1, n2)


@jax.jit
def tag_gene(n1: jnp.ndarray, n2: jnp.ndarray) -> float:
    assert n1.shape == n2.shape
    n2_1 = n2[0]
    diff = (n1[1:] - n2_1)
    return jnp.sum(a_v(diff) * jnp.exp(- jnp.abs(diff)))
