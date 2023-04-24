import jax.random as jrd
import jax.numpy as jnp
from jax import lax
import gymnax
import flax.linen as nn

from gene.network import LinearModel
from gene.utils import genome_size

from gene.distances import jit_vmap_distance_f


# TODO: remove below and wrap into _genome_to_model_func and partially apply evaluate_individual
# L2 dist vmap over 2 axis (returns matrix) and jit
_jit_vmap_L2_dist = jit_vmap_distance_f("L2")
# tag dist vmap over 2 axis (returns matrix) and jit
_jit_vmap_tag_dist = jit_vmap_distance_f("tag")


def _genome_to_model(_genome: list[float], settings: dict):
    assert _genome.shape[0] == genome_size(settings)
    layer_dims = settings["net"]["layer_dimensions"]
    d = settings["d"]

    split_i = sum(layer_dims) * d

    # To facilitate acces to the encoding of the weights and the biases (and reduce confusion and possible error in computing indexes), we split the genome in 2 parts
    _genome_w, _genome_b = jnp.split(_genome, [split_i])

    model_parameters = {}
    for i, (layer_in, layer_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        # Split the genome into subarrays, each subarray is the position vector for one neuron
        genome_w_positions = jnp.array(jnp.split(_genome_w, sum(layer_dims)))

        layer_offset = sum(layer_dims[:i])
        # indexes of the previous layer neurons
        src_idx = layer_offset + jnp.arange(start=0, stop=layer_in)
        # indexes of the current layer neurons
        target_idx = layer_offset + layer_in + jnp.arange(start=0, stop=layer_out)

        weight_matrix = _jit_vmap_L2_dist(genome_w_positions, src_idx, target_idx)
        # Biases are directly encoded into the genome, they are stored at the end of the genome, in _genome_b
        biases = lax.dynamic_slice(
            _genome_b, (sum(layer_dims[1 : i + 1]),), (layer_out,)
        )

        model_parameters[f"Dense_{i}"] = {
            "kernel": weight_matrix,
            "bias": biases,
        }

    # To parameter FrozenDict
    model = LinearModel(layer_dims[1:])
    model_parameters = nn.FrozenDict({"params": model_parameters})
    return model, model_parameters


def _rollout_problem_lax(
    model: nn.Module,
    model_parameters: dict,
    rng: jrd.KeyArray,
    settings: dict,
):
    """Perform a complete rollout of the current env, declared in 'settings', for a specific individual and returns the cumulated reward as the fitness"""
    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(settings["problem"]["environnment"])
    obs, state = env.reset(rng_reset, env_params)

    def rollout_loop(val):
        obs, state, done, rng, cum_reward = val

        rng, rng_step = jrd.split(rng, 2)
        action = jnp.argmax(model.apply(model_parameters, obs))
        n_obs, n_state, reward, done, _ = env.step(rng_step, state, action, env_params)

        new_val = n_obs, n_state, done, rng, cum_reward + reward
        return new_val

    val = lax.while_loop(
        lambda val: jnp.logical_not(val[2]), rollout_loop, (obs, state, False, rng, 0)
    )
    _, _, _, _, cum_reward = val

    return cum_reward


def evaluate_individual(
    genome: jnp.array,
    rng: jrd.KeyArray,
    settings: dict,
):
    # Decodes the genome into the model parameters
    model, model_parameters = _genome_to_model(genome, settings=settings)
    # Perform the evaluation step
    fitness = _rollout_problem_lax(
        model=model,
        model_parameters=model_parameters,
        rng=rng,
        settings=settings,
    )

    return fitness
