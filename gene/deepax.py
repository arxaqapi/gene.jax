import jax.numpy as jnp
import jax.random as jrd
import jax
from functools import partial

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

class ReLU(Modulax):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return jnp.maximum(x, 0)


class Sequential(Modulax):
    def __init__(self, *args) -> None:
        self.layers = args

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

class Linear(Modulax):
    def __init__(self, in_features: jnp.ndarray, out_features: jnp.ndarray, parameters: dict = None, rng: Generator = None) -> None:
        super().__init__()
        # Init if parameters dict is empty
        if parameters is None:
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            k = 1 / in_features
            subkey_W, subkey_b = rng.create_subkeys(2)
            self.parameters = {
                'w': jax.random.uniform(subkey_W, shape=(in_features, out_features), minval=-k, maxval=k),
                'b': jax.random.uniform(subkey_b, shape=(out_features,), minval=-k, maxval=k)
            }
        else:
            self.parameters = parameters
        assert self.parameters['w'].shape == (in_features, out_features)
        assert self.parameters['b'].shape == (out_features, )

    def forward(self, x):
        """
        - x in R^(batch, input_dims)
        - W in R^(input_dims, output_dims)
        """
        self.input_cache = x
        return x @ self.parameters['w'] + self.parameters['b']


class FlatNet(Modulax):
    """Simple static linear neural network"""
    def __init__(self, genome: jnp.ndarray = None, layer_dimensions: list[int] = [128, 64, 64, 18], distance_f = None) -> None:
        """
        128 input nodes
        L1 64 nodes
        L2 64 nodes
        18 output nodes -> distribution over action space
        """
        self.layer_dimensions = layer_dimensions
        self.distance_f = distance_f
        # Init each layer from the genome
        # https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static
        parameters = _from_genome_to_model(genome, d=3)
        self.model = Sequential(
            Linear(in_features=parameters[0]['in'], out_features=parameters[0]['out'], parameters= {
                'w': jnp.reshape(parameters[0]['w'], newshape=(parameters[0]['in'], parameters[0]['out'])),
                'b': parameters[0]['b']
            }),
            ReLU(),
            Linear(in_features=parameters[1]['in'], out_features=parameters[1]['out'], parameters= {
                'w': jnp.reshape(parameters[1]['w'], newshape=(parameters[1]['in'], parameters[1]['out'])),
                'b': parameters[1]['b']
            }),
            ReLU(),
            Linear(in_features=parameters[2]['in'], out_features=parameters[2]['out'], parameters= {
                'w': jnp.reshape(parameters[2]['w'], newshape=(parameters[2]['in'], parameters[2]['out'])),
                'b': parameters[2]['b']
            }),
        )

    def forward(self, x) -> jnp.ndarray:
        return self.model(x)
    

@partial(jax.jit, static_argnames=['d'])
def _from_genome_to_model(genome: jnp.ndarray, d: int = 3):
    # SECTION: Temporary
    distance_f = tag_gene
    layer_dimensions = [128, 64, 64, 18]
    # !SECTION: Temporary
    assert genome.shape[0] == sum(layer_dimensions) * (d + 1)
    # value = 0
    parameters = [{
        'in': None,
        'out': None,
        'w': jnp.zeros((in_ * out, )),
        'b': jnp.zeros((out, ))} for in_, out in zip(layer_dimensions[:-1], layer_dimensions[1:])]
    # layers = []
    for i, (layer_in, layer_out) in enumerate(zip(layer_dimensions[:-1], layer_dimensions[1:])):
        parameters[i]['in'] = layer_in
        parameters[i]['out'] = layer_out
        # ==========================================================================================
        # ================ Weights =================================================================
        w_index = 0
        for start_n in range(sum(layer_dimensions[:i]) * (d + 1), sum(layer_dimensions[:i + 1]) * (d + 1), d + 1):
            for end_n in range(sum(layer_dimensions[:i + 1]) * (d + 1), sum(layer_dimensions[:i + 2]) * (d + 1), d + 1):
                # NOTE: we only use dimension d for the neurons positions, the leftover value is for the bias
                neurons_positions = genome[start_n : start_n + d], genome[end_n : end_n + d]
                weight = distance_f(*neurons_positions)
                # NOTE: mutating values
                parameters[i]['w'] = parameters[i]['w'].at[w_index].set(weight)
                w_index += 1
                # DEBUG
                # value += 1
        # ================= Biases =================================================================
        for b_i, bias_position in enumerate(range(sum(layer_dimensions[:i + 1]) * (d + 1), sum(layer_dimensions[:i + 2]) * (d + 1), d + 1)):
            parameters[i]['b'] = parameters[i]['b'].at[b_i].set(genome[bias_position + d])
        # f_best, x_best = jax.lax.fori_loop(0, lam, lambda i, fx_b: opl_eval(x, N[i, :], fx_b[0], fx_b[1]), (f_best, x_best))
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html
        # jax.lax.fori_loop(
        #     sum(layer_dimensions[:i + 1]) * (d + 1),
        #     sum(layer_dimensions[:i + 2]),  # * (d + 1) / (d + 1)  # NOTE A vérifier
        #     lambda b_i, v: v.at[b_i].set(genome[b_i * (d + 1) + d]),  # bias_position = b_i * (d + 1)  # NOTE: a vérif
        #     parameters[i]['b']
        # )
        # ==========================================================================================
    exit(14)
    return parameters
