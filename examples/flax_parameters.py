"""Simple Linear model to show how the parameters of a flax module are passed 
as arguments (nothing is stored inside flax module instances) 
- https://flax.readthedocs.io/en/latest/guides/flax_basics.html
"""

from typing import Sequence

import jax.numpy as jnp
import jax.random as jrd
import jax
import flax.linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

if __name__ == '__main__':
    model = MLP([10, 8, 2])  # 3 couches: (12, 10) (10, 8) (8, 2)
    batch = jnp.ones((32, 12))  # Batch, Features
    # model_parameters = model.init(jrd.PRNGKey(0), batch)
    # output = model.apply(model_parameters, batch)


    # print(jax.tree_util.tree_map(lambda x: x.shape, model_parameters))
    # variables['params']['Dense_0']['kernel']
    # variables['params']['Dense_0']['bias']

    model_parameters = nn.FrozenDict({'params': {
        'Dense_0': {
            'kernel': jnp.ones((12, 10)),
            'bias': jnp.ones((10,))
        },
        'Dense_1': {
            'kernel': jnp.ones((10, 8)),
            'bias': jnp.ones((8,))
        },
        'Dense_2': {
            'kernel': jnp.ones((8, 2)),
            'bias': jnp.ones((2,))
        }
    }})
    output = model.apply(model_parameters, batch)

