import jax.numpy as jnp
from jax import jit, vmap
import chex
import flax.linen as nn

from functools import partial


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
    """Introduced in [A comparison of genetic regulatory network dynamics and encoding](https://doi.org/10.1145/3071178.3071322) ''"""
    # raise ValueError("Does not work as expected")
    n1 = x[n1_i]
    n2 = x[n2_i]

    diff = n1[1:] - n2[0]
    return jnp.sum(_a(diff) * jnp.exp(-jnp.abs(diff)))


def pL2_dist(x, n1_i, n2_i):
    """The product L2 distance, is simply the L2 distance multiplied by the bounded prouduct
    of all components of the vector. This is not a mathematically strict "distance"
    function, since it allows for negative values."""
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


@chex.dataclass
class NonLinearDistance:
    model: nn.Module
    model_parameters: nn.FrozenDict

    @partial(jit, static_argnums=(0))
    def _apply(self, x) -> float:
        return self.model.apply(self.model_parameters, x)

    def nn_distance(self, x, n1_i, n2_i):
        return self._apply(jnp.concatenate(x[n1_i], x[n2_i]))


Distances = {"L2": L2_dist, "tag": tag_dist, "pL2": pL2_dist}

Vectorized_distances = {k: jit_vmap_distance_f(k) for k, _v in Distances.items()}
