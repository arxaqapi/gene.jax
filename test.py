from brax import envs
from brax.envs.wrappers import EpisodeWrapper
import chex
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jrd
import jax

rng = jrd.PRNGKey(0)
rng, rng_reset = jrd.split(rng, 2)

# define a halfcheeta env
env = envs.get_environment(env_name="halfcheetah")
env = EpisodeWrapper(env, 500, 1)


state = env.reset(rng_reset)

# randomly sample action
for i in range(10):
    rng, rng_act = jrd.split(rng, 2)
    # FIXME: problem seems to be here
    actions = jrd.normal(rng_act, shape=(6, ))
    # step
    n_state = env.step(state, actions)

    # look at reward
    jax.debug.print("[Debug]: {reward}", reward=state.reward)

    state = n_state

print('End')