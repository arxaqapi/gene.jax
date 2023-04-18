import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym

from stafe.deepax import FlatNet


def evaluate_models(env, models):
    return jnp.array([evaluate_model(env, model=model) for model in models])


def evaluate_model(env, model):
    # https://gymnasium.farama.org/environments/atari/space_invaders/
    input_obs_shape = sum(env.observation_space.shape)
    output_act_space_shape = env.action_space.n
    assert 128 == input_obs_shape
    assert 18 == output_act_space_shape
    # Stats
    stats = {
        'rewards': []}

    observation, info = env.reset()
    terminated, truncated = False, False
    while (not terminated) and (not truncated):
        # NOTE: normalize observations
        action_distr = model(observation / 255)
        a = np.argmax(action_distr)  # env.action_space.sample()

        # print(f'[DEBUG] - {a}')
        observation, reward, terminated, truncated, _ = env.step(a)
        stats['rewards'].append(reward)
    
    return np.sum(stats['rewards'])


def evaluate_flatnet(env, genome):
    model = FlatNet(genome)
    return evaluate_model(env=env, model=model)