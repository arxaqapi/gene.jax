import jax.random as jrd
import jax.numpy as jnp
from jax import lax, jit
import flax.linen as nn
import gymnax
from brax.v1 import envs
from brax.v1.envs.wrappers import EpisodeWrapper  # brax.envs.wrapper


# ================================================
# ===============   Gymnax   =====================
# ================================================


def rollout_gymnax_task(
    model: nn.Module,
    model_parameters: nn.FrozenDict,
    rng: jrd.KeyArray,
    config: dict,
):
    """Perform a complete rollout of the current env, declared in 'config',
    for a specific individual and returns the cumulated reward as the fitness.
    """
    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(config["task"]["environnment"])
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


# ================================================
# ================   Brax   ======================
# ================================================


def get_brax_env(config: dict):
    env = envs.get_environment(env_name=config["task"]["environnment"])
    return EpisodeWrapper(
        env, episode_length=config["task"]["episode_length"], action_repeat=1
    )


# TODO - make carry a dictionnary
def rollout_brax_task(
    config: dict,
    model: nn.Module,
    model_parameters: nn.FrozenDict,
    env,
    rng_reset: jrd.KeyArray,
) -> float:
    state = jit(env.reset)(rng_reset)

    def rollout_loop(carry, x):
        env_state, cum_reward = carry
        actions = model.apply(model_parameters, env_state.obs)
        new_state = jit(env.step)(env_state, actions)

        corrected_reward = new_state.reward * (1 - new_state.done)
        new_carry = new_state, cum_reward + corrected_reward
        return new_carry, None

    carry, _ = lax.scan(
        f=rollout_loop,
        init=(state, state.reward),
        xs=None,
        length=config["task"]["episode_length"],
    )

    return carry[-1]
