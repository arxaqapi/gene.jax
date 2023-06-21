import jax.numpy as jnp
import jax.random as jrd
from jax.tree_util import tree_map
from jax import jit, lax
from gymnax.visualize import Visualizer
import gymnax

import time
from pathlib import Path

from gene.v1.evaluate import genome_to_model, get_brax_env


def run_env(genome: jnp.ndarray, config: dict, rng: jrd.KeyArray = jrd.PRNGKey(0)):
    model, model_parameters = genome_to_model(genome, config)

    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(config["task"]["environnment"])
    obs, state = env.reset(rng_reset, env_params)

    state_seq, reward_seq = [], []
    done = False
    while not done:
        # NOTE: Stats
        state_seq.append(state)

        rng, rng_step = jrd.split(rng, 2)
        # Sample a random action.
        action = jnp.argmax(model.apply(model_parameters, obs))

        # Perform the step transition.
        n_obs, n_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        # NOTE: Stats
        reward_seq.append(reward)

        obs = n_obs
        state = n_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))

    loc = Path(f"animations/{int(time.time())}__cartpole_r_{cum_rewards[-1]}.gif")
    loc.parent.mkdir(parents=True, exist_ok=True)
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate(str(loc), view=False)

    return loc


def visualize_brax(config: dict, genome, rng: jrd.KeyArray = jrd.PRNGKey(0)):
    # 1. genome_to_model
    model, model_parameters = genome_to_model(genome, config)

    # 2. get env
    env = get_brax_env(config)

    # 3. rollout w. static evaluation function
    base_state = jit(env.reset)(rng)

    def f(carry, x: None):
        cur_state = carry

        actions = model.apply(model_parameters, cur_state.obs)
        new_state = jit(env.step)(cur_state, actions)

        new_carry = new_state
        return new_carry, cur_state.pipeline_state

    _carry, pipeline_states = lax.scan(
        f=f,
        init=base_state,
        xs=None,
        length=config["task"]["episode_length"],
    )

    flat_pipeline_states = [
        tree_map(lambda x: x[i], pipeline_states)
        for i in range(config["task"]["episode_length"])
    ]

    return env, flat_pipeline_states
