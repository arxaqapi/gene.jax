"""
https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html
"""
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm


xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise

# plt.scatter(xs, ys)
# plt.show()

def model(theta, x):
    """Computes xw + b on a batch of input data x."""
    w, b = theta
    return w * x + b

# Loss function is (y - y^)^2
def loss_fn(theta, x, y):
    return  jnp.mean(jnp.square(model(theta, x) - y))

# theta = theta - lr * Grad Loss(y, y^)
def update(theta, x, y, lr: float = 0.1):
    return theta - lr * jax.grad(loss_fn)(theta, x, y)

# Training loop
theta = jnp.array([1., 1.])
for _ in tqdm(range(1000)):
    theta = update(theta, xs, ys)

w, b = theta

plt.scatter(xs, ys)
plt.plot(xs, model(theta, xs), color="r")
plt.title(f'Linear regression: {w=}, {b=}')
plt.show()


