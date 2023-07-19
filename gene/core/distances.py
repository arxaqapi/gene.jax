import pickle
from pathlib import Path
from functools import partial
from typing import Type

import flax.linen as nn
import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.tree_util import register_pytree_node_class

from gene.core import decoding, models


@jit
def _L2_dist(v1, v2):
    diff = v1 - v2
    return jnp.sqrt(diff.dot(diff))


@jit
def _a(x):
    x = jnp.where(x > 1, 1, x)
    x = jnp.where(x < -1, -1, x)
    return x


class DistanceFunction:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def distance(self, v1: Array, v2: Array) -> float:
        """Distance computation between two vectors. Need to be reimplemented.

        Args:
            v1 (Array): Position vector of neuron 1
            v2 (Array): Position vector of neuron 2

        Raises:
            NotImplementedError: _description_

        Returns:
            float: distance value between these two neurons.
        """
        raise NotImplementedError

    def measure(self, genotype_weights: Array, gene_1: int, gene_2: int) -> float:
        """Measures the distance between `genotype_weights[gene_1]`
        and `genotype_weights[gene_2]`

        Args:
            genotype_weights (Array): Array of weight vectors, representing the
                position of the neurons in an n-dimensional space.
            gene_1 (int): index of the first vector in `genotype_weights`
            gene_2 (int): index of the second vector in `genotype_weights`

        Returns:
            float: Distance value measured between 2 points in an n-dimensional space.
        """
        return self.distance(genotype_weights[gene_1], genotype_weights[gene_2])

    @property
    def vectorized_measure(self):
        if not hasattr(self, "_f"):
            self._f = jit(
                vmap(
                    vmap(self.measure, in_axes=(None, None, 0)), in_axes=(None, 0, None)
                ),
            )
        return self._f


class pL2Distance(DistanceFunction):
    def __init__(self) -> None:
        super().__init__()

    @partial(jit, static_argnums=(0))
    def distance(self, v1: Array, v2: Array) -> float:
        diff = v1 - v2
        return _a(jnp.prod(diff)) * _L2_dist(v1, v2)


# TODO - finish implementation, add tests
@register_pytree_node_class
class NNDistance(DistanceFunction):
    def __init__(
        self,
        distance_genome: Array,
        config: dict,
        nn_layers_dims: list[int] = [6, 32, 32, 1],
    ) -> None:
        """Initializes a neural network parametrized distance function,
        that takes an input vector representing two neuron positions concatenated
        adn returns a distance measure between them.

        Args:
            distance_genome (Array): genome of the distance function to decoded
                into the phenotype (neural net)
            config (dict): _description_
            nn_layers_dims (list[int]): layer dimensions of the neural network
        """
        super().__init__()
        self.distance_genome = distance_genome
        self.config = config
        self.nn_layers_dims = nn_layers_dims

        self.model_parameters = decoding.DirectDecoder(config).decode(distance_genome)
        self.model: nn.Module = models.LinearModel(nn_layers_dims[1:])

    @partial(jit, static_argnums=(0))
    def distance(self, v1: Array, v2: Array) -> float:
        """Computes the neural-network parametrized distance between
        vectors `v1` and `v2`"""
        return self.model.apply(self.model_parameters, jnp.concatenate((v1, v2)))

    def save_parameters(self, path: Path) -> None:
        """Saves the `model_parameters` to `path`.

        Args:
            path (Path): The Path and name of the file where it will be saved
        """
        with path.open("wb") as f:
            pickle.dump(self.model_parameters, f)

    def load_parameters(self, path: Path) -> dict:
        """Load the saved `model_parameters` from `path`
        to the model_parameters attribute.

        Args:
            path (Path): The Path and name of the file to retrieve.

        Returns:
            dict: state dictionnary of the loaded model parameters.
        """
        state = {}
        with path.open("rb") as f:
            self.model_parameters = pickle.load(f)
        return state

    # Pytree methods
    def tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {
            "distance_genome": self.distance_genome,
            "config": self.config,
            "nn_layers_dims": self.nn_layers_dims,
            "model_parameters": self.model_parameters,
            "model": self.model,
        }  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            aux_data["distance_genome"], aux_data["config"], aux_data["nn_layers_dims"]
        )


@register_pytree_node_class
class NNDistanceSimple(DistanceFunction):
    def __init__(
        self,
        model_parameters,
        model,
    ) -> None:
        """Dead simple neural-network based distance function"""
        super().__init__()
        self.model_parameters = model_parameters
        self.model: nn.Module = model

    @partial(jit, static_argnums=(0))
    def distance(self, v1: Array, v2: Array) -> float:
        """Computes the neural-network parametrized distance between
        vectors `v1` and `v2`"""
        return self.model.apply(self.model_parameters, jnp.concatenate((v1, v2)))

    def save_parameters(self, path: Path) -> None:
        """Saves the `model_parameters` to `path`.

        Args:
            path (Path): The Path and name of the file where it will be saved
        """
        with path.open("wb") as f:
            pickle.dump(self.model_parameters, f)

    def load_parameters(self, path: Path) -> dict:
        """Load the saved `model_parameters` from `path`
        to the model_parameters attribute.

        Args:
            path (Path): The Path and name of the file to retrieve.

        Returns:
            dict: state dictionnary of the loaded model parameters.
        """
        state = {}
        with path.open("rb") as f:
            self.model_parameters = pickle.load(f)
        return state

    # Pytree methods
    def tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {
            "model_parameters": self.model_parameters,
            "model": self.model,
        }  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


class CGPDistance(DistanceFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raise NotImplementedError


Distance_functions = {"pL2": pL2Distance, "nn": NNDistance, "cgp": CGPDistance}


def get_df(config: dict) -> Type[DistanceFunction]:
    return Distance_functions[config["encoding"]["distance"]]
