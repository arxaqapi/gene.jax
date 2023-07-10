import jax.random as jrd
import jax.numpy as jnp
from jax import lax, jit
import flax.linen as nn
import gymnax


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


# TODO - rename to get_braxv1_env
def get_brax_env(config: dict):
    from brax.v1 import envs as envs_v1
    from brax.v1.envs.wrappers import EpisodeWrapper as EpisodeWrapper_v1

    env = envs_v1.get_environment(env_name=config["task"]["environnment"])
    return EpisodeWrapper_v1(
        env, episode_length=config["task"]["episode_length"], action_repeat=1
    )


def get_braxv2_env(config: dict):
    from brax import envs as envs_v2
    from brax.envs.wrapper import EpisodeWrapper as EpisodeWrapper_v2

    env = envs_v2.get_environment(env_name=config["task"]["environnment"])
    return EpisodeWrapper_v2(
        env, episode_length=config["task"]["episode_length"], action_repeat=1
    )


def rollout_brax_task(
    config: dict,
    model: nn.Module,
    model_parameters: nn.FrozenDict,
    env,
    rng_reset: jrd.KeyArray,
) -> float:
    """Perform a single rollout of the environnment
    with the given policy parametrized by a `model` and its `model_parameters`.
    The return (sum of all the undiscounted rewards) is used as a fitness value.

    Args:
        config (dict): config of the current run.
        model (nn.Module): model used as a policy `f: S -> A`
        model_parameters (nn.FrozenDict): parameters of the model
        env (_type_): environnment used to perform the evaluation
        rng_reset (jrd.KeyArray): rng key used for evaluation

    Returns:
        float: return used as a fitness value.
    """
    state = jit(env.reset)(rng_reset)

    def rollout_loop(carry, x):
        env_state, cumulative_reward, active_episode = carry
        actions = model.apply(model_parameters, env_state.obs)
        new_state = jit(env.step)(env_state, actions)

        corrected_reward = new_state.reward * active_episode
        cumulative_reward = cumulative_reward + corrected_reward
        # we carry over the active_episode value.
        # If it becomes 0, it stays at 0 because of the multiplication
        # this desactivated the reward counting in cumulative_reward by setting
        # the corrected_reward to 0
        new_active_episode = active_episode * (1 - new_state.done)
        return new_state, cumulative_reward, new_active_episode, None

    active_episode = jnp.ones_like(state.reward)
    carry, _ = lax.scan(
        f=rollout_loop,
        init=(state, state.reward, active_episode),
        xs=None,
        length=config["task"]["episode_length"],
    )

    return carry[-1]
