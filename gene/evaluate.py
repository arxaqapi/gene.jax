import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap, lax
import jax
import gymnax
import flax.linen as nn

from gene.network import LinearModel


def _L2_dist(x, base, target_offset, d: int):
    diff = lax.dynamic_slice(x, (base,), (d,)) - lax.dynamic_slice(
        x, (target_offset,), (d,)
    )
    return jnp.sqrt(diff.dot(diff))


_vmap_L2_dist = vmap(_L2_dist, in_axes=(None, None, 0, None))
_vvmap_L2_dist = vmap(_vmap_L2_dist, in_axes=(None, 0, None, None))
_jitted_L2_dist = jit(_vvmap_L2_dist, static_argnames=["d"])


def _genome_to_model(genome: list[float], settings: dict):
    layer_dimensions = settings["net"]["layer_dimensions"]
    assert genome.shape[0] == sum(layer_dimensions)

    # FIXME: Testing without biases for the moment
    model_parameters = {}
    for i, (layer_in, layer_out) in enumerate(
        zip(layer_dimensions[:-1], layer_dimensions[1:])
    ):
        position_offset = sum(layer_dimensions[:i])
        # indexes of the previous layer neurons
        src_idx = position_offset + jnp.arange(
            start=0, stop=layer_in, step=settings["d"]
        )
        # indexes of the current layer neurons
        target_idx = (
            position_offset
            + layer_in
            + jnp.arange(start=0, stop=layer_out, step=settings["d"])
        )

        weight_matrix = _jitted_L2_dist(genome, src_idx, target_idx, settings["d"])
        model_parameters[f"Dense_{i}"] = {
            "kernel": weight_matrix,
            "bias": jnp.zeros((layer_out,)),
        }
        # print(model_parameters[f"Dense_{i}"]['kernel'].shape)


    # To parameter FrozenDict
    model = LinearModel(layer_dimensions[1:])
    model_parameters = nn.FrozenDict({"params": model_parameters})
    # print(jax.tree_util.tree_map(lambda x: x.shape, model_parameters))
    # exit(10)
    return model, model_parameters


def _rollout_problem(
    model: nn.Module, model_parameters: dict, rng: jrd.KeyArray, settings: dict
):
    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(settings["problem"]["environnment"])
    obs, state = env.reset(rng_reset, env_params)

    cum_reward = 0
    done = False
    while not done:
        # Update RNG
        rng, rng_step = jrd.split(rng, 2)
        # Sample a random action.
        action_disrtibution = model.apply(model_parameters, obs)
        action = jnp.argmax(action_disrtibution)

        # Perform the step transition.
        n_obs, n_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        # Update Stats
        cum_reward += reward

        obs = n_obs
        state = n_state

    return cum_reward


def evaluate_individual(
    genome: jnp.array,
    rng: jrd.KeyArray,
    settings: dict,
):
    # Genome to model
    model, model_parameters = _genome_to_model(genome, settings=settings)
    # run_rollout
    fitness = _rollout_problem(
        model=model,
        model_parameters=model_parameters,
        rng=rng,
        settings=settings,
    )
    return fitness
