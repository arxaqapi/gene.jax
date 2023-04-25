import jax.numpy as jnp
from jax import jit, vmap


@jit
def _L2_dist(v1, v2):
    diff = v1 - v2
    return jnp.sqrt(diff.dot(diff))


@jit
def _a(x):
    x = jnp.where(x > 1, 1, x)
    x = jnp.where(x < -1, -1, x)
    return x


# Distance functions


def L2_dist(x, n1_i, n2_i):
    return _L2_dist(x[n1_i], x[n2_i])


def tag_dist(x, n1_i, n2_i):
    raise ValueError("Does not work as expected")
    n1 = x[n1_i]
    n2 = x[n2_i]
    n2_1 = n2[0]
    diff = n1[1:] - n2_1
    return jnp.sum(_a(diff) * jnp.exp(-jnp.abs(diff)))


def pL2_dist(x, n1_i, n2_i):
    diff = x[n1_i] - x[n2_i]
    return _a(jnp.prod(diff)) * _L2_dist(x[n1_i], x[n2_i])


def jit_vmap_distance_f(distance_f: str):
    """Takes the name of a the distance function and optimizes it (vmap over 2 axis)
    to run in parallel over the complete genome and to return a matrix of the correct
    shape: (in_features, out_features).
    """
    return jit(
        vmap(
            vmap(Distances[distance_f], in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        )
    )


Distances = {"L2": L2_dist, "tag": tag_dist, "pL2": pL2_dist}
