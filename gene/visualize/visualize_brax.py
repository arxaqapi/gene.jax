"""
Utility code to visualize brax learned policies on
the given task.
"""
import jax.random as jrd
import flax.linen as nn
from jax import jit, lax, tree_util, Array
from brax.v1.io import html

from gene.core.evaluation import get_brax_env
from gene.core.decoding import Decoder, Decoders
from gene.core.distances import DistanceFunction


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
    env = get_brax_env(config)

    # 3. rollout w. static evaluation function
    base_state = jit(env.reset)(rng)

    if use_v1:

        def f(carry, x: None):
            cur_state = carry

            actions = model.apply(model_parameters, cur_state.obs)
            new_state = jit(env.step)(cur_state, actions)

            new_carry = new_state
            # Brax v1, state.qp
            return new_carry, cur_state.qp

    else:

        def f(carry, x: None):
            cur_state = carry

            actions = model.apply(model_parameters, cur_state.obs)
            new_state = jit(env.step)(cur_state, actions)

            new_carry = new_state
            # Brax v2, state.pipeline_state
            return new_carry, cur_state.pipeline_state

    _, pipeline_states = lax.scan(
        f=f,
        init=base_state,
        xs=None,
        length=config["task"]["episode_length"],
    )

    flat_pipeline_states = [
        tree_util.tree_map(lambda x: x[i], pipeline_states)
        for i in range(config["task"]["episode_length"])
    ]

    return env, flat_pipeline_states


def render_brax(env, pipeline_states: list, output_file: str = ""):
    html_string = html.render(
        env.sys.replace(dt=env.dt), pipeline_states, height="100vh", colab=False
    )

    with open(f"{output_file}.html", "w") as f:
        f.writelines(html_string)
