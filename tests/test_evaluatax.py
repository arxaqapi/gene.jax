import unittest
import jax
from stafe.evaluatax import evaluate_model
from stafe.deepax import FlatNet
import gymnasium as gym


class TestEvluation(unittest.TestCase):
    def test_evaluation_full_net(self):
        env = gym.make('ALE/SpaceInvaders-v5', obs_type='ram', full_action_space=True)
        model = FlatNet(jax.random.uniform(jax.random.PRNGKey(2), (1096, ), minval=-10, maxval=10))
        total_reward = evaluate_model(env, model)
        print(f'Reward: {total_reward}')

        env.close() 

        self.assertGreaterEqual(total_reward, 0.)