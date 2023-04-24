import jax.random as jrd
import jax.numpy as jnp
from jax import jit, vmap, lax
import jax
import gymnax
import flax.linen as nn

from gene.network import LinearModel
from gene.utils import genome_size


def _L2_dist(x, n1_i, n2_i):
    diff = x[n1_i] - x[n2_i]
    return jnp.sqrt(diff.dot(diff))


@jit
def _a(x):
    x = jnp.where(x > 1, 1, x)
    x = jnp.where(x < -1, -1, x)
    # x = jnp.maximum(x, -1)
    # x = jnp.minimum(x, 1)
    return x


def _tag_dist(x, n1_i, n2_i):
    n1 = x[n1_i]
    n2 = x[n2_i]
    n2_1 = n2[0]
    diff = n1[1:] - n2_1
    return jnp.sum(_a(diff) * jnp.exp(-jnp.abs(diff)))


# L2 dist vmap over 2 axis (returns matrix) and jit
_jit_vmap_L2_dist = jit(
    vmap(vmap(_L2_dist, in_axes=(None, None, 0)), in_axes=(None, 0, None))
)

# tag dist vmap over 2 axis (returns matrix) and jit
_jit_vmap_tag_dist = jit(
    vmap(vmap(_tag_dist, in_axes=(None, None, 0)), in_axes=(None, 0, None))
)


def _genome_to_model(_genome: list[float], settings: dict):
    assert _genome.shape[0] == genome_size(settings)
    layer_dims = settings["net"]["layer_dimensions"]
    d = settings["d"]

    # FIXME: Testing without biases for the moment
    split_i = sum(layer_dims) * d
    _genome_w, _genome_b = jnp.split(_genome, [split_i])

    model_parameters = {}
    for i, (layer_in, layer_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        # ==== Weights / Kernel
        # genome_w_positions = jnp.split(_genome_w, sum(layer_dims))
        genome_w_positions = jnp.array(jnp.split(_genome_w, sum(layer_dims)))

        layer_offset = sum(layer_dims[:i])
        # indexes of the previous layer neurons
        src_idx = layer_offset + jnp.arange(start=0, stop=layer_in)
        # indexes of the current layer neurons
        target_idx = layer_offset + layer_in + jnp.arange(start=0, stop=layer_out)

        weight_matrix = _jit_vmap_L2_dist(genome_w_positions, src_idx, target_idx)
        # Biases are directly encoded into the genome
        biases = lax.dynamic_slice(
            _genome_b, (sum(layer_dims[1 : i + 1]),), (layer_out,)
        )

        model_parameters[f"Dense_{i}"] = {
            "kernel": weight_matrix,
            "bias": biases,
        }
    # ==== Biases

    # To parameter FrozenDict
    model = LinearModel(layer_dims[1:])
    model_parameters = nn.FrozenDict({"params": model_parameters})
    # print(jax.tree_util.tree_map(lambda x: x.shape, model_parameters))
    return model, model_parameters


def _rollout_problem_lax(
    model: nn.Module,
    model_parameters: dict,
    rng: jrd.KeyArray,
    settings: dict,
):
    # Init model
    # model = LinearModel(settings["net"]["layer_dimensions"][1:])
    # model_parameters = nn.FrozenDict(model_parameters)

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


# def _rollout_problem(
#     model: nn.Module, model_parameters: dict, rng: jrd.KeyArray, settings: dict
# ):
#     rng, rng_reset = jrd.split(rng, 2)

#     env, env_params = gymnax.make(settings["problem"]["environnment"])
#     obs, state = env.reset(rng_reset, env_params)

#     cum_reward = 0
#     done = False
#     while not done:
#         # Update RNG
#         rng, rng_step = jrd.split(rng, 2)
#         # Sample a random action.
#         action_disrtibution = model.apply(model_parameters, obs)
#         action = jnp.argmax(action_disrtibution)

#         # Perform the step transition.
#         n_obs, n_state, reward, done, _ = env.step(rng_step, state, action, env_params)
#         # Update Stats
#         cum_reward += reward

#         obs = n_obs
#         state = n_state

#     return cum_reward


def evaluate_individual(
    genome: jnp.array,
    rng: jrd.KeyArray,
    settings: dict,
):
    # jit_genome_to_model = jit(partial(_genome_to_model, settings=settings))
    # model_parameters = jit_genome_to_model(
    #     genome
    # )

    # Genome to model
    model, model_parameters = _genome_to_model(genome, settings=settings)
    # run_rollout
    fitness = _rollout_problem_lax(
        model=model,
        model_parameters=model_parameters,
        rng=rng,
        settings=settings,
    )

    return fitness
