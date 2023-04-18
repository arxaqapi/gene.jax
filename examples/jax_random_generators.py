import jax

# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng
key = jax.random.PRNGKey(0)

for i in range(10):
    # Use old key to generate subkey and overwrite old key
    key, subkey = jax.random.split(key)
    # Use the subkey to generate data
    random_data = jax.random.normal(subkey, shape=(1,))
    print(random_data)
