"""
Utility code to visualize brax learned policies on
the given task.
"""
from pathlib import Path

import jax.random as jrd
import flax.linen as nn
from jax import jit, lax, tree_util, Array

from gene.core.evaluation import get_braxv1_env, get_braxv2_env
from gene.core.decoding import Decoder, Decoders
from gene.core.distances import DistanceFunction


# TODO - replace model with otional dependency and get it from config file
def visualize_brax(
    config: dict,
    genome: Array,
    model: nn.Module,
    df: DistanceFunction,
    rng: jrd.KeyArray = jrd.PRNGKey(0),
    use_v1: bool = True,
):
    # 1. Decode genome
    decoder: Decoder = Decoders[config["encoding"]["type"]](config, df)
    model_parameters = decoder.decode(genome)

    # 2. get env
    env = get_braxv1_env(config) if use_v1 else get_braxv2_env(config)

    # 3. rollout w. static evaluation function
    base_state = jit(env.reset)(rng)

    if use_v1:

        def f(carry, x: None):
            cur_state, r = carry

            actions = model.apply(model_parameters, cur_state.obs)
            new_state = jit(env.step)(cur_state, actions)

            new_carry = new_state, r + new_state.reward
            # Brax v1, state.qp
            return new_carry, cur_state.qp

    else:

        def f(carry, x: None):
            cur_state, r = carry

            actions = model.apply(model_parameters, cur_state.obs)
            new_state = jit(env.step)(cur_state, actions)

            new_carry = new_state, r + new_state.reward
            # Brax v2, state.pipeline_state
            return new_carry, cur_state.pipeline_state

    _carry, pipeline_states = lax.scan(
        f=f,
        init=(base_state, base_state.reward),
        xs=None,
        length=config["task"]["episode_length"],
    )

    # NOTE - to remove
    print(f"Fitness: {_carry[-1]}")

    flat_pipeline_states = [
        tree_util.tree_map(lambda x: x[i], pipeline_states)
        for i in range(config["task"]["episode_length"])
    ]

    return env, flat_pipeline_states, use_v1


def render_brax(output_file: Path, env, pipeline_states: list, use_v1: bool = True):
    if use_v1:
        from brax.v1.io import html as html_v1

        html_string = html_v1.render(
            env.sys,
            pipeline_states,
            height="100",
        )
    else:
        from brax.io import html as html_v2

        html_string = html_v2.render(
            env.sys.replace(dt=env.dt), pipeline_states, height="100", colab=False
        )

    html_string = html_string.replace("height: 100px;", "height: 100vh;")

    out = Path(output_file).with_suffix(".html")

    with out.open("w") as f:
        f.writelines(html_string)
