import jax.random as jrd
import jax.numpy as jnp
from jax import lax, jit
from flax.linen import Module
import gymnax

from functools import partial

from gene.encoding import genome_to_model


def _rollout_problem_lax(
    model: Module,
    model_parameters: dict,
    rng: jrd.KeyArray,
    config: dict,
):
    """Perform a complete rollout of the current env, declared in 'config', for a specific individual and returns the cumulated reward as the fitness"""
    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(config["problem"]["environnment"])
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


# Brax loops
def _rollout_brax(
    config: dict, model, model_parameters, env, rng_reset: jrd.KeyArray
) -> float:
    state = jit(env.reset)(rng_reset)

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
        length=config["problem"]["episode_length"],
    )
    # chex.assert_trees_all_close(carry[-1], jnp.cumsum(_rewards)[-1])

    return carry[-1]  # Access cumuluative reward aka. return


@partial(jit, static_argnums=(1, 2, 3))
def evaluate_individual_brax(
    genome: jnp.array,
    rng: jrd.KeyArray,
    config: dict,
    env,
) -> float:
    model, model_parameters = genome_to_model(genome, config=config)

    fitness = _rollout_brax(
        model=model,
        model_parameters=model_parameters,
        config=config,
        env=env,
        rng_reset=rng,
    )
    return fitness
