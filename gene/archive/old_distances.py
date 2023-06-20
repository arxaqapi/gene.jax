# from functools import partial

import jax.numpy as jnp
from jax import jit, vmap  # , Array

# import chex
# import flax.linen as nn

# from gene.encoding import _direct_decoding
# from gene.network import LinearModel


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
    """Introduced in [A comparison of genetic regulatory network dynamics and encoding](https://doi.org/10.1145/3071178.3071322)"""
    # raise ValueError("Does not work as expected")
    n1 = x[n1_i]
    n2 = x[n2_i]

    diff = n1[1:] - n2[0]
    return jnp.sum(_a(diff) * jnp.exp(-jnp.abs(diff)))


def pL2_dist(x, n1_i, n2_i):
    """The product L2 distance, is simply the L2 distance multiplied by the
    bounded prouduct of all components of the vector.
    This is not a mathematically strict _distance_ function, since it allows
    for negative values.
    """
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

Vectorized_distances = {k: jit_vmap_distance_f(k) for k, _v in Distances.items()}


# ====================================================================================
# ============================ Experimental ==========================================
# ====================================================================================

# FIXME - Circular imports
# @chex.dataclass
# class NNDistance:
#     """Neural Network Distance function.

#     This distance function uses a neural network to compute
#     the distance between two vectors.
#     """

#     distance_genome: Array
#     layer_dimensions: tuple[int]

#     def __post_init__(self):
#         model, model_parameters = self._genome_to_nn_distance_model(
#             self.distance_genome, self.layer_dimensions
#         )
#         self.model: nn.Module = model
#         self.model_parameters: nn.FrozenDict = model_parameters

#     @partial(jit, static_argnums=(0))
#     def _apply(self, x) -> float:
#         return self.model.apply(self.model_parameters, x)

#     def nn_distance(self, x, n1_i, n2_i):
#         return self._apply(jnp.concatenate(x[n1_i], x[n2_i]))

#     @property
#     def vmap_evaluate(self):
#         """Returns the vectorized version of `nn_distance` and caches it.
#         Can then be used to retrieve the vectorized function"""
#         # Create vectorized function if not already created
#         if not hasattr(self, "_f"):
#             self._f = jit(
#                 vmap(
#                     vmap(self.nn_distance, in_axes=(None, None, 0)),
#                     in_axes=(None, 0, None),
#                 )
#             )
#         return self._f

#     def _genome_to_nn_distance_model(
#         distance_genome: jnp.ndarray, layer_dimensions: list[int]
#     ):
#         # layer_dims are static

#         model_parameters = _direct_decoding(distance_genome, layer_dimensions)
#         model = LinearModel(layer_dimensions[1:])
#         return model, nn.FrozenDict({"params": model_parameters})
