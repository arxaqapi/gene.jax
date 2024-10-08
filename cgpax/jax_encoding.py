from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

from cgpax.jax_functions import function_switch, constants


@jit
def __offset_copy__(
    idx: int, carry: Tuple[int, jnp.ndarray, jnp.ndarray]
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    offset, source, buffer = carry
    buffer = buffer.at[idx + offset].set(source.at[idx].get())
    return offset, source, buffer


@jit
def __update_buffer__(
    buffer_idx: int, carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_genes, y_genes, f_genes, buffer = carry
    n_in = len(buffer) - len(x_genes)
    idx = buffer_idx - n_in
    f_idx = f_genes.at[idx].get()
    x_arg = buffer.at[x_genes.at[idx].get()].get()
    y_arg = buffer.at[y_genes.at[idx].get()].get()

    buffer = buffer.at[buffer_idx].set(function_switch(f_idx, x_arg, y_arg))
    return x_genes, y_genes, f_genes, buffer


@jit
def __update_register__(
    row_idx: int,
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
    lhs_genes, x_genes, y_genes, f_genes, n_in, register = carry
    lhs_idx = lhs_genes.at[row_idx].get() + n_in
    f_idx = f_genes.at[row_idx].get()
    x_idx = x_genes.at[row_idx].get()
    x_arg = register.at[x_idx].get()
    y_idx = y_genes.at[row_idx].get()
    y_arg = register.at[y_idx].get()
    register = register.at[lhs_idx].set(function_switch(f_idx, x_arg, y_arg))
    return lhs_genes, x_genes, y_genes, f_genes, n_in, register


def genome_to_cgp_program(
    genome: jnp.ndarray,
    config: dict,
    outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh,
) -> Callable:
    """Transforms a genome of a cgp programm into an actual program
    that can be evaluated on an array of inputs and a buffer"""
    n_in_env = config["n_in_env"]
    n_const = config["n_constants"]
    n_in = config["n_in"]
    n_nodes = config["n_nodes"]

    x_genes, y_genes, f_genes, out_genes = jnp.split(
        genome, [n_nodes, 2 * n_nodes, 3 * n_nodes]
    )

    def program(inputs: jnp.ndarray, buffer: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        # copy actual inputs
        _, _, buffer = fori_loop(0, n_in_env, __offset_copy__, (0, inputs, buffer))
        # copy constants
        _, _, buffer = fori_loop(
            0, n_const, __offset_copy__, (n_in_env, constants, buffer)
        )

        _, _, _, buffer = fori_loop(
            n_in, len(buffer), __update_buffer__, (x_genes, y_genes, f_genes, buffer)
        )
        outputs = jnp.take(buffer, out_genes)
        bounded_outputs = outputs_wrapper(outputs)

        return buffer, bounded_outputs

    return jit(program)


def genome_to_lgp_program(
    genome: jnp.ndarray,
    config: dict,
    outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh,
) -> Callable:
    n_in_env = config["n_in_env"]
    n_const = config["n_constants"]
    n_in = config["n_in"]
    n_out = config["n_out"]
    n_rows = config["n_rows"]
    n_registers = config["n_registers"]
    output_positions = jnp.arange(start=n_registers - n_out, stop=n_registers)

    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)

    def program(
        inputs: jnp.ndarray, register: jnp.ndarray
    ) -> (jnp.ndarray, jnp.ndarray):
        register = jnp.zeros(n_registers)
        # copy actual environment inputs
        _, _, register = fori_loop(0, n_in_env, __offset_copy__, (0, inputs, register))
        # copy constant inputs
        _, _, register = fori_loop(
            0, n_const, __offset_copy__, (n_in_env, constants, register)
        )
        _, _, _, _, _, register = fori_loop(
            0,
            n_rows,
            __update_register__,
            (lhs_genes, x_genes, y_genes, f_genes, n_in, register),
        )
        outputs = jnp.take(register, output_positions)
        bounded_outputs = outputs_wrapper(outputs)

        return register, bounded_outputs

    return jit(program)
