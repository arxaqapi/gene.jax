import jax.random as jrd
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, jit
import flax.linen as nn
import gymnax
from brax.v1 import envs  # brax
from brax.v1.envs.wrappers import EpisodeWrapper  # brax.envs.wrapper

from gene.v1.encoding import genome_to_model


# ================================================
# ===============   Gymnax   =====================
# ================================================


def _rollout_problem_lax(
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


def evaluate_individual(
    genome: jnp.array,
    rng: jrd.KeyArray,
    config: dict,
):
    # Decodes the genome into the model parameters
    model, model_parameters = genome_to_model(genome, config=config)
    # Perform the evaluation step
    fitness = _rollout_problem_lax(
        model=model,
        model_parameters=model_parameters,
        rng=rng,
        config=config,
    )

    return fitness


# ================================================
# ================   Brax   ======================
# ================================================


def get_brax_env(config: dict):
    env = envs.get_environment(env_name=config["task"]["environnment"])
    return EpisodeWrapper(
        env, episode_length=config["task"]["episode_length"], action_repeat=1
    )


def _rollout_brax(
    config: dict, model, model_parameters, env, rng_reset: jrd.KeyArray
) -> float:
    state = jit(env.reset)(rng_reset)

    # jax.debug.print("[🤯] Observation vector: {x}", x=state.obs.shape)

    def rollout_loop(carry, x):
        # FIXME: carry could be reduced
        env_state, cum_reward = carry
        actions = model.apply(model_parameters, env_state.obs)
        new_state = jit(env.step)(env_state, actions)

        corrected_reward = new_state.reward * (1 - new_state.done)
        new_carry = new_state, cum_reward + corrected_reward
        return new_carry, corrected_reward

    carry, _rewards = lax.scan(
        f=rollout_loop,
        init=(state, state.reward),
        xs=None,
        length=config["task"]["episode_length"],
    )
    # chex.assert_trees_all_close(carry[-1], jnp.cumsum(_rewards)[-1])

    return carry[-1]  # Access cumuluative reward aka. return


def evaluate_individual_brax(
    genome: jnp.array,
    rng: jrd.KeyArray,
    config: dict,
    env,
) -> float:
    model, model_parameters = genome_to_model(genome, config=config)

    # from jax import tree_util
    # print(tree_util.tree_map(lambda x: x.shape, model_parameters))

    fitness = _rollout_brax(
        model=model,
        model_parameters=model_parameters,
        config=config,
        env=env,
        rng_reset=rng,
    )
    return fitness


# ================================================
# =================   Soft   =====================
# ================================================
# print rewards (since goal is still to max rewards even if we scale down by prob)


# NOTE - temp
def _rollout_soft(
    model: nn.Module,
    model_parameters: nn.FrozenDict,
    rng: jrd.KeyArray,
    rng_action_sampling: jrd.KeyArray,
    config: dict,
):
    """Soft rollout of the current policy defined as a neural network
    on the environment defined in the config dict.

    Args:
        model (nn.Module): _description_
        model_parameters (dict): _description_
        rng (jrd.KeyArray): _description_
        rng_action_sampling (jrd.KeyArray): _description_
        config (dict): _description_

    Returns:
        _type_: _description_
    """
    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(config["task"]["environnment"])
    obs, state = env.reset(rng_reset, env_params)
    action_space = jnp.array(range(env.action_space().n))  # [0, 1]

    def rollout_loop(carry, x):
        obs, state, done, rng, _cum_reward = carry

        rng, rng_step = jrd.split(rng, 2)

        logits = model.apply(model_parameters, obs)
        action_distr = jnn.softmax(logits)
        action = jrd.choice(rng_action_sampling, a=action_space, p=action_distr)

        n_obs, n_state, reward, done, _ = env.step(rng_step, state, action, env_params)

        corrected_reward = reward * (1 - done)
        new_carry = n_obs, n_state, done, rng, _cum_reward + reward
        return new_carry, (
            corrected_reward,
            action_distr[action],
        )  # carry, y being appended to ys

    carry, stacked_r_distr = lax.scan(
        f=rollout_loop,
        init=(obs, state, False, rng, 0),
        xs=None,
        length=config["task"]["episode_length"],
    )
    # compute backwards fitness value
    fitness = 0
    for r, pi in zip(reversed(stacked_r_distr[0]), reversed(stacked_r_distr[1])):
        fitness = pi * (fitness + r)

    return fitness


# NOTE - temp
def evaluate_individual_soft(
    genome: jnp.array,
    rng: jrd.KeyArray,
    rng_action_sampling: jrd.KeyArray,
    config: dict,
):
    """La valeur est pas représentative de la "vrai" fitness."""
    # Decodes the genome into the model parameters
    model, model_parameters = genome_to_model(genome, config=config)
    # Perform the evaluation step
    fitness = _rollout_soft(
        model=model,
        model_parameters=model_parameters,
        rng=rng,
        rng_action_sampling=rng_action_sampling,
        config=config,
    )

    return fitness
