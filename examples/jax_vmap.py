import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap, lax

from functools import partial


# https://ericmjl.github.io/dl-workshop/02-jax-idioms/01-loopless-loops.html#exercise-chained-vmaps
# https://github.com/google/jax/discussions/5199
# https://github.com/google/jax/issues/673
# https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap

# https://github.com/google/jax/pull/13096


@jit
def mini_test(x, base, target_offset):
    return jnp.sqrt(jnp.square(x[base] - x[10 + target_offset]))


def run_vvmap_test():
    vmap_mini_test = vmap(mini_test, in_axes=(None, None, 0))
    vvmap_mini_test = vmap(vmap_mini_test, in_axes=(None, 0, None))

    x = jnp.arange(10, 24)
    offsets = jnp.arange(start=0, stop=4, step=1)
    base_i = jnp.arange(start=0, stop=10, step=1)

    res = vvmap_mini_test(x, base_i, offsets)
    print(res)


# ============================================================================


def L2_dist(x, base, target_offset, d: int):
    diff = lax.dynamic_slice(x, (base,), (d,)) - lax.dynamic_slice(
        x, (target_offset,), (d,)
    )
    return jnp.sqrt(diff.dot(diff))


vmap_L2_dist = vmap(L2_dist, in_axes=(None, None, 0, None))
vvmap_L2_dist = vmap(vmap_L2_dist, in_axes=(None, 0, None, None))
jitted_L2_dist = jit(vvmap_L2_dist, static_argnames=["d"])


# @partial(jit, static_argnames=['d', 'layer_dimensions'])
# @partial(jit, static_argnames=['d'])
def genome_to_param(genome: jnp.ndarray, d: int, layer_dimensions: list[int]):
    # layer_dimensions = [784, 64, 128, 10]
    assert genome.shape[0] == sum(layer_dimensions)
    # NOTE: Testing without biases for the moment
    parameters = []
    for i, (layer_in, layer_out) in enumerate(
        zip(layer_dimensions[:-1], layer_dimensions[1:])
    ):
        position_offset = sum(layer_dimensions[:i])
        src_idx = position_offset + jnp.arange(
            start=0, stop=layer_in, step=d
        )  # indexes of the previous layer neurons
        target_idx = (
            position_offset + layer_in + jnp.arange(start=0, stop=layer_out, step=d)
        )  # indexes of the current layer neurons

        weight_matrix = jitted_L2_dist(genome, src_idx, target_idx, d)
        parameters.append(
            {"w": weight_matrix},
        )

    return parameters


# ======================================================
def test_large_net():
    layer_dimensions = [784, 64, 128, 10]
    genome = jnp.zeros((sum(layer_dimensions),))
    genome = genome.at[784 + 64 + 3].set(20)
    genome = genome.at[784 + 64 + 128 + 5].set(10)

    parameters = genome_to_param(genome, d=1)

    assert parameters[0]["w"].shape == (784, 64)
    assert parameters[1]["w"].shape == (64, 128)
    assert parameters[2]["w"].shape == (128, 10)

    assert parameters[2]["w"][3, 5] == jnp.sqrt(
        jnp.sum(jnp.square(genome[784 + 64 + 3] - genome[784 + 64 + 128 + 5]))
    )

    assert parameters[2]["w"][3, 6] == jnp.sqrt(
        jnp.sum(jnp.square(genome[784 + 64 + 3] - genome[784 + 64 + 128 + 6]))
    )
    assert parameters[2]["w"][3, :].sum() == 190


def map_over_genomes():
    layer_dimensions = [784, 64, 128, 10]
    BATCH_SIZE_OR_POP_SIZE = 1000
    genomes = jnp.zeros(
        (
            BATCH_SIZE_OR_POP_SIZE,
            sum(layer_dimensions),
        )
    )

    v_genomes_to_param = jit(
        vmap(partial(genome_to_param, d=1, layer_dimensions=layer_dimensions))
    )

    parameters = v_genomes_to_param(genomes)

    # print(parameters)
    print(parameters[0]["w"].shape)  # (10, 784, 64)
    print(parameters[1]["w"].shape)  # (10, 64, 128)
    print(parameters[2]["w"].shape)  # (10, 128, 10)
    # print(len(parameters))


if __name__ == "__main__":
    # test_large_net()
    map_over_genomes()
