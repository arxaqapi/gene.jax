import jax.random as jrd
import jax.numpy as jnp
from jax import lax
from flax.linen import Module
import gymnax


from gene.encoding import genome_to_model


def _rollout_problem_lax(
    model: Module,
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
    model, model_parameters = genome_to_model(genome, settings=settings)
    # Perform the evaluation step
    fitness = _rollout_problem_lax(
        model=model,
        model_parameters=model_parameters,
        rng=rng,
        settings=settings,
    )

    return fitness
