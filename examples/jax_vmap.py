import jax.numpy as jnp
import jax.random as jrd
import jax


# https://ericmjl.github.io/dl-workshop/02-jax-idioms/01-loopless-loops.html#exercise-chained-vmaps
# https://github.com/google/jax/discussions/5199
# https://github.com/google/jax/issues/673
# https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap

@jax.jit
def mini_test(x, base, target_offset):
    return jnp.sqrt(jnp.square(x[base] - x[10 + target_offset]))

def run_vvmap_test():
    vmap_mini_test = jax.vmap(mini_test, in_axes=(None, None, 0))
    vvmap_mini_test = jax.vmap(vmap_mini_test, in_axes=(None, 0, None))

    x = jnp.arange(10, 24)
    offsets = jnp.arange(start=0, stop=4, step=1)
    base_i = jnp.arange(start=0, stop=10, step=1)

    res = vvmap_mini_test(x, base_i, offsets)
    print(res)


# ============================================================================

@jax.jit
def L2_dist(x, base, target_offset):
    """
    target_offset (list[int]): the target index in the 
    """
    # x should not change, only base and target_offset are changing 
    return jnp.sqrt(jnp.square(x[base] - x[target_offset]))


vmap_L2_dist = jax.vmap(L2_dist, in_axes=(None, None, 0))
vvmap_L2_dist = jax.vmap(vmap_L2_dist, in_axes=(None, 0, None))


def genome_to_param(genome: jnp.ndarray, d: int = 1, layer_dimensions: list = [10, 4]):
    assert genome.shape[0] == (10 + 4)
    # NOTE: Testing without biases for the moment
    # pos = 0  # FIXME: to be used to fix position over multiples layers
    for i, (layer_in, layer_out) in enumerate(zip(layer_dimensions[:-1], layer_dimensions[1:])):
        # 0, (10, 4)
        # FIXME: not complete, accumulate offset !!!
        src_idx = jnp.arange(start=0, stop=layer_in, step=d)  # indexes of the previous layer neurons
        target_idx = layer_in + jnp.arange(start=0, stop=layer_out, step=d)  # indexes of the current layer neurons

        weight_matrix = vvmap_L2_dist(genome, src_idx, target_idx)

        return weight_matrix


res = genome_to_param(
    jnp.arange(10, 24),
)
print(res)
assert res.shape == (10, 4)

# raise ValueError('Not tested!!')
