import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym

from gene.deepax import FlatNet, SmallFlatNet, MountainFlatNet

# def evaluate_models(env, models):
#     return jnp.array([evaluate_model(env, model=model) for model in models])


# def evaluate_model(env, model):
#     # https://gymnasium.farama.org/environments/atari/space_invaders/
#     input_obs_shape = sum(env.observation_space.shape)
#     output_act_space_shape = env.action_space.n
#     assert 128 == input_obs_shape
#     assert 18 == output_act_space_shape
#     # Stats
#     stats = {
#         'rewards': []}

#     observation, info = env.reset()
#     terminated, truncated = False, False
#     while (not terminated) and (not truncated):
#         # NOTE: normalize observations
#         action_distr = model(observation / 255)
#         a = np.argmax(action_distr)  # env.action_space.sample()

#         # print(f'[DEBUG] - {a}')
#         observation, reward, terminated, truncated, _ = env.step(a)
#         stats['rewards'].append(reward)
    
#     return np.sum(stats['rewards'])


# def evaluate_flatnet(env, genome):
#     model = FlatNet(genome)
#     return evaluate_model(env=env, model=model)


def _evaluate_small_model(genome):
    model = SmallFlatNet()
    model.init(genome)
    env = gym.make('ALE/SpaceInvaders-v5', obs_type='ram', full_action_space=True)

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
    env.close()
    return np.sum(stats['rewards'])


import gymnax

def eval_mountain_car(genome):
    print(genome)
    model = MountainFlatNet()
    model.init(genome)

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_step = jax.random.split(rng, 3)

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("MountainCar-v0")
    # Reset the environment.
    obs, state = env.reset(key_reset, env_params)
    
    assert 2 == env.observation_space.shape[0]
    assert 3 == env.num_actions
    # Stats
    cum_ret = 0
    done = False
    while not done:
        # Model guided action selection
        action_distr = model(obs)
        a = np.argmax(action_distr) 
        # Perform the step transition.
        n_obs, n_state, reward, done, _ = env.step(key_step, state, a, env_params)
        # Reward
        cum_ret += reward

        obs = n_obs
        state = n_state

    return cum_ret



# evaluate_model_vmap = jax.vmap(_evaluate_small_model, in_axes=0) 
evaluate_model_vmap = jax.vmap(eval_mountain_car, in_axes=0) 


# ===================================================
# Brax 


# def evaluate_brax(genome):
#     model = FlatNet(genome)
#     # model
#     pass