"""
Loss Landscape Analysis code.

- Create the grid to interpolate onto.
- Vmap over all points and evaluate interpolated genomes
"""
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap, lax
from brax import envs
from brax.envs.wrapper import EpisodeWrapper

from pathlib import Path
from functools import partial

from evaluate import genome_to_model


# =============================================================
# =============================================================

def rollout_brax(
    config: dict, model, model_parameters, env, rng_reset
) -> float:
    state = jit(env.reset)(rng_reset)

    def rollout_loop(carry, x):
        env_state, cum_reward = carry
        actions = model.apply(model_parameters, env_state.obs)
        new_state = jit(env.step)(env_state, actions)

        corrected_reward = new_state.reward * (1 - new_state.done)
        new_carry = new_state, cum_reward + corrected_reward
        return new_carry, corrected_reward

    carry, _ = lax.scan(
        f=rollout_loop,
        init=(state, state.reward),
        xs=None,
        length=config["problem"]["episode_length"],
    )
    # chex.assert_trees_all_close(carry[-1], jnp.cumsum(returns)[-1])

    return carry[-1]


@partial(jit, static_argnums=(1, 2, 3))
def evaluate_individual_brax(
    genome: jnp.array,
    rng: jrd.KeyArray,
    config: dict,
    env,
) -> float:
    model, model_parameters = genome_to_model(genome, config=config)

    fitness = rollout_brax(
        model=model,
        model_parameters=model_parameters,
        config=config,
        env=env,
        rng_reset=rng,
    )
    return fitness
# =============================================================
# =============================================================
# =============================================================


def get_env(config: dict):
    env = envs.get_environment(env_name=config["problem"]["environnment"])
    return  EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )



def load_genomes(path_initial: Path, path_final: Path) -> tuple[jax.Array, jax.Array]:
    with open(path_initial, "rb") as f:
        initial_genome = jnp.load(f)
    with open(path_final, "rb") as f:
        final_genome = jnp.load(f)
    return initial_genome, final_genome


# FIXME - correct eval func (haflcheetah one)
def evaluate_all(
    genomes: list[jax.Array], rng: jrd.KeyArray, config: dict, env
) -> list[float]:
    """Evaluate all individual in the list of genomes

    Args:
        genomes (list[jax.Array]): list of the genomes of all individuals to evaluate
        rng (jrd.KeyArray): rng key used to run the simulation
        config (dict): config of the current run

    Returns:
        list[float]: List of floats corresponding to the fitness value per individual
    """
    return [evaluate_individual_brax(genome, rng, config, env) for genome in genomes]


def interpolate_2D(
    initial_genome: jax.Array,
    final_genome: jax.Array,
    n: int,
    key: jrd.KeyArray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Performs 2D interpolation between the initial and final network.

    - .. _An empirical analysis of the optimization of deep network loss surfaces: http://arxiv.org/abs/1612.04010.
    - .. _Visualizing the Loss Landscape of Neural Nets: http://arxiv.org/abs/1712.09913.


    Args:
        initial_genome (jax.Array): The genome of the initial individual's neural network.
        final_genome (jax.Array): The genome of the final individual's neural network.
        n (int): number of points to interpolate per axis.
        key (jrd.KeyArray): key used to generate the random vector.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: genomes, x, y
    """
    # taken from https://github.com/TemplierPaul/QDax/blob/main/analysis/landscape_analysis_2d.py
    v1 = final_genome - initial_genome

    # NOTE - Gram-Schmidt Process ??
    # https://www.ucl.ac.uk/~ucahmdl/LessonPlans/Lesson10.pdf
    v2 = jax.random.normal(key, shape=v1.shape)
    v2 = v2 - jnp.dot(v2, v1) * v1 / jnp.dot(v1, v1)
    v2 = v2 / jnp.linalg.norm(v2) * jnp.linalg.norm(v1)

    x, y = jnp.meshgrid(jnp.linspace(-1, 2, n), jnp.linspace(-1, 2, n))
    _x = x.reshape((-1, 1))
    _y = y.reshape((-1, 1))

    # See equation 1 in http://arxiv.org/abs/1712.09913
    genomes = _x * v1 + _y * v2 + initial_genome
    return genomes, x, y


def plot_ll(
    values: jax.Array,
    X: jax.Array,
    Y: jax.Array,
    initial_genome: jax.Array,
    initial_genome_fitness: float,
    final_genome: jax.Array,
    final_genome_fitness: float,
    export_name: str = "test",
) -> None:
    """Loss Landscape plotting function. Exports the final plot as a png and an interactive html file using plotly.

    Args:
        values (jax.Array): Computed fitness values for each interpolated genomes
        X (jax.Array): X values of the interpolated genomes
        Y (jax.Array): Y values of the interpolated genomes
        initial_genome (jax.Array): genomes of the initial individual. Starting point of the search.
        final_genome (jax.Array): genomes of the final individual. End point of the search.
        export_name (str, optional): Name of the output visualization files. Defaults to "test".
    """
    import plotly.graph_objects as go

    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=values)])

    x_initial, y_initial, z_initial = *initial_genome, initial_genome_fitness
    x_final, y_final, z_final = *final_genome, final_genome_fitness
    fig.add_scatter3d(
        name="Initial genome",
        x=(x_initial,),
        y=(y_initial,),
        z=(z_initial,),
        legendrank=1,
    )
    fig.add_scatter3d(
        name="Final genome", x=(x_final,), y=(y_final,), z=(z_final,), legendrank=1
    )

    fig.update_layout(
        title="3D plot test",
        # https://plotly.com/python/3d-camera-controls/
        scene_camera={
            "up": dict(x=0, y=0, z=1),
            "center": dict(x=0, y=0, z=0),
            "eye": dict(x=-1.5, y=1.5, z=1.5),
        },
        # https://stackoverflow.com/questions/61827165/plotly-how-to-handle-overlapping-colorbar-and-legends
        # legend_orientation="h",
        # https://plotly.com/python/legend/
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(f"{export_name}.html")
    fig.write_image(f"{export_name}.png")


# TODO: finish me
def lla(rng: jrd.KeyArray = jrd.PRNGKey(0)):
    rng, interpolation_rng, eval_rng = jrd.split(rng, 3)

    # NOTE - 1. downlad files from run
    # https://docs.wandb.ai/guides/track/public-api-guide#download-a-file-from-a-run
    import wandb

    api = wandb.Api()
    run = api.run("arxaqapi/Brax halfcheetah/7r08z3mz")
    config = run.config

    env = get_env(config)

    # run = api.run("<entity>/Brax halfcheetah/floral-water-42")
    path_initial = (
        run.file("genomes/1684939081_g0_mean_indiv.npy").download(replace=True).name
    )
    path_final = (
        run.file("genomes/1684939081_g121_mean_indiv.npy").download(replace=True).name
    )

    # NOTE - 2. load files
    initial_genome, final_genome = load_genomes(path_initial, path_final)
    # NOTE - 3. interpolate
    genomes, xs, ys = interpolate_2D(
        initial_genome, final_genome, n=10, key=interpolation_rng
    )
    # NOTE - 4. evaluate at each interpolation step
    # FIXME - correct eval func (haflcheetah one)
    values = evaluate_all(genomes, rng=eval_rng, config=config, env=env)
    # NOTE - 5. plot landscape
    plot_ll(
        values,
        xs,
        ys,
        initial_genome,
        evaluate_individual_brax(initial_genome, eval_rng, config, env),
        final_genome,
        evaluate_individual_brax(final_genome, eval_rng, config, env),
    )


if __name__ == "__main__":
    lla()
