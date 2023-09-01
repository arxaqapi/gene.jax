"""
Neural network properties evaluation
"""
import flax.linen as nn
from flax import traverse_util
import jax.numpy as jnp
import jax.random as jrd


def expressivity_ratio(model_parameters: nn.FrozenDict):
    """Ratio of unique parameters in the network.
    Number of unique parameters over the total number of parameters

    - Close to 0: not a lot of unique params
    - Close to 1: lots of unique params
    """
    flat_model_param = traverse_util.flatten_dict(model_parameters, sep=".")

    unique = set()
    total_size = 0
    for _, layer_params in flat_model_param.items():
        total_size += layer_params.size

        for param in jnp.ravel(layer_params):
            unique.add(param.astype(float))

    assert total_size > 0
    return len(unique) / total_size


def initialization_term(model_parameters: nn.FrozenDict):
    """Distance de la moyenne de la distribution des paramètres avec 0
    et écart-type de la distribution."""
    flat_model_param = traverse_util.flatten_dict(model_parameters, sep=".")
    param_vector = jnp.concatenate(
        [jnp.ravel(params) for params in flat_model_param.values()]
    )

    mean = jnp.mean(param_vector)
    std = jnp.std(param_vector)

    return mean, std


def input_distribution_restoration(
    model: nn.Module,
    model_parameters: nn.FrozenDict,
    batch_size: int = 1,
    rng_gen: jrd.KeyArray = jrd.PRNGKey(0),
):
    """Ability of a network to output the same distribution as the input one."""
    model_parm_values = list(model_parameters["params"].values())
    in_size: int = model_parm_values[0]["kernel"].shape[0]
    # out_size: int = model_parm_values[-1]["kernel"].shape[-1]

    mu, sigma = 0.0, 1.0
    x = mu + jrd.normal(rng_gen, shape=(batch_size, in_size)) * sigma
    y = model.apply(model_parameters, x)

    y_mean = jnp.mean(y)
    y_std = jnp.std(y)

    return (mu, sigma), (y_mean, y_std)
