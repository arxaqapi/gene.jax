import jax.numpy as jnp
import jax.random as jrd
from gymnax.visualize import Visualizer
import gymnax

import time
from pathlib import Path

from gene.evaluate import genome_to_model


def run_env(genome: jnp.ndarray, config: dict, rng: jrd.KeyArray = jrd.PRNGKey(0)):
    model, model_parameters = genome_to_model(genome, config)

    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(config["problem"]["environnment"])
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
