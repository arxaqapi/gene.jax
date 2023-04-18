import jax.numpy as jnp
import numpy as np
import jax.random as jrd
import jax
from functools import partial
from tqdm import tqdm

from gene.utilax import backend
from gene.distances import tag_gene

assert backend() == "gpu"


class Generator:
    def __init__(self, base_seed: int = 0) -> None:
        self.base_key = jax.random.PRNGKey(base_seed)

    def __call__(self) -> jax.random.KeyArray:
        return self.create_subkeys()

    def create_subkeys(self, num: int = 1) -> jax.random.KeyArray:
        self.base_key, *subkeys = jax.random.split(self.base_key, num + 1)
        return subkeys


class Modulax:
    def __init__(self) -> None:
        self.input_cache = None
        self.type = None

    def __call__(self, *args) -> jnp.ndarray:
        """Alias for forward, convenience function.

        Args:
           args: The inputs, e.g., the output of the previous layer.
        """
        return self.forward(*args)

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class Activation(Modulax):
    def __init__(self) -> None:
        super().__init__()
        self.type = "activation"


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return jnp.maximum(x, 0)


class Sequential(Modulax):
    def __init__(self, *args, initialized: bool = False) -> None:
        self.layers: list[Linear] = args
        self.initialized = initialized

    def __call__(self, *args) -> jnp.ndarray:
        """Alias for forward, convenience function.

        Args:
           args: The inputs, e.g., the output of the previous layer.
        """
        return self.forward(*args)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_param_from_genome(self, genome, d: int = 3):
        self.layer_dimensions = [self.layers[0].in_features] + [l.out_features for l in self.layers if l.type != 'activation']
        # SECTION - Get parameters from genome
        partial_genome_to_model_parameters = partial(_genome_to_model_parameters, layer_dimensions=self.layer_dimensions, d=d)
        parameters = jax.jit(partial_genome_to_model_parameters)(genome=genome)
        # !SECTION

        i = 0
        for layer in self.layers:
            if layer.type != 'activation':
                layer.parameters['w'] = jnp.reshape(parameters[i]['w'], (layer.in_features, layer.out_features))
                layer.parameters['b'] = parameters[i]['b']
                i += 1
    def count_neurons(self):
        # if self.layer_dimensions is not None:
        return sum([self.layers[0].in_features] + [l.out_features for l in self.layers if l.type != 'activation'])



class Linear(Modulax):
    def __init__(self, in_features: jnp.ndarray, out_features: jnp.ndarray, parameters: dict = None) -> None: # initialized: bool = False
        super().__init__()
        self.type = "linear"
        self.in_features = in_features
        self.out_features = out_features

        if parameters is None:
            self.parameters = {
                'w': None,
                'b': None,}
        else:
            self.parameters = parameters
        
    def forward(self, x):
        """
        - x in R^(batch, input_dims)
        - W in R^(input_dims, output_dims)
        """
        # self.input_cache = x  # NOTE: Reduce memory consumption
        assert self.parameters['w'].shape == (self.in_features, self.out_features)
        assert self.parameters['b'].shape == (self.out_features, )
        return x @ self.parameters['w'] + self.parameters['b']


class FlatNet(Modulax):
    """Simple static linear neural network"""
    def __init__(self) -> None:
        """
        128 input nodes
        L1 64 nodes
        L2 64 nodes
        18 output nodes -> distribution over action space
        """
        # https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static
        self.model = Sequential(
            Linear(in_features=128, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=18),)

    def init(self, genome: jnp.ndarray = None, d: int = 1):
        self.model.init_param_from_genome(genome, d=d)

    def forward(self, x) -> jnp.ndarray:
        return self.model(x)


class SmallFlatNet(Modulax):
    """Simple static linear neural network"""
    def __init__(self) -> None:
        """
        128 input nodes
        18 output nodes -> distribution over action space
        """
        # https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static
        self.model = Sequential(
            Linear(in_features=128, out_features=18),
            ReLU(),)

    def init(self, genome: jnp.ndarray = None, d: int = 1):
        self.model.init_param_from_genome(genome, d=d)

    def forward(self, x) -> jnp.ndarray:
        return self.model(x)

    def neurons(self) -> int:
        # assert 128 * 18 == self.model.count_neurons()
        return self.model.count_neurons()


class MountainFlatNet(Modulax):
    """Simple static linear neural network"""
    def __init__(self) -> None:
        """
        128 input nodes
        18 output nodes -> distribution over action space
        """
        # https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static
        self.model = Sequential(
            Linear(in_features=2, out_features=10),
            ReLU(),
            Linear(in_features=10, out_features=3),
            ReLU(),)

    def init(self, genome: jnp.ndarray = None, d: int = 2):
        self.model.init_param_from_genome(genome, d=d)

    def forward(self, x) -> jnp.ndarray:
        return self.model(x)

    def neurons(self) -> int:
        assert 2 + 10 + 3 == self.model.count_neurons()
        return self.model.count_neurons()


# @partial(jax.jit, static_argnames=['d'])
def _genome_to_model_parameters(genome: jnp.ndarray, layer_dimensions: np.ndarray, d: int = 3):  # list[int]
    """Take a genome and automatically constructs the parameter dict {'w': [], 'b': []} list"""

    parameter_list = [{
        'w': jnp.zeros((in_ * out, )),
        'b': jnp.zeros((out, ))} for in_, out in zip(layer_dimensions[:-1], layer_dimensions[1:])
    ]

    # for i, (layer_in, layer_out) in enumerate(tqdm(zip(layer_dimensions[:-1], layer_dimensions[1:]), desc='genome split', position=1, leave=False)):
    for i, (layer_in, layer_out) in enumerate(zip(layer_dimensions[:-1], layer_dimensions[1:])):
        w_index = 0
        for start_n in range(sum(layer_dimensions[:i]) * (d + 1), sum(layer_dimensions[:i + 1]) * (d + 1), d + 1):
            for end_n in range(sum(layer_dimensions[:i + 1]) * (d + 1), sum(layer_dimensions[:i + 2]) * (d + 1), d + 1):
                # NOTE: we only use dimension d for the neurons positions, the leftover value is for the bias
                weight = tag_gene(genome[start_n : start_n + d], genome[end_n : end_n + d])
                # NOTE: mutating values
                parameter_list[i]['w'] = parameter_list[i]['w'].at[w_index].set(weight)
                w_index += 1
        # ================= Biases =================================================================
        for b_i, bias_position in enumerate(range(sum(layer_dimensions[:i + 1]) * (d + 1), sum(layer_dimensions[:i + 2]) * (d + 1), d + 1)):
            parameter_list[i]['b'] = parameter_list[i]['b'].at[b_i].set(genome[bias_position + d])

    return parameter_list