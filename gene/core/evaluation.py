from functools import partial

import jax.random as jrd
import jax.numpy as jnp
from jax import lax, jit, vmap
import flax.linen as nn
import gymnax

from gene.core.distances import NNDistance, CGPDistance
from gene.core.decoding import GENEDecoder
from gene.core.models import get_model
from gene.nn_properties import (
    expressivity_ratio,
    weights_distribution,
    input_distribution_restoration,
)

# ================================================
# ===============   Gymnax   =====================
# ================================================


def rollout_gymnax_task(
    model: nn.Module,
    model_parameters: nn.FrozenDict,
    rng: jrd.KeyArray,
    config: dict,
) -> float:
    """Perform a complete rollout of the current env, declared in 'config',
    for a specific individual and returns the cumulated reward as the fitness.
    """
    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make(config["task"]["environnment"])
    obs, state = env.reset(rng_reset, env_params)

    def rollout_loop(val):
        obs, state, done, rng, cum_reward = val

        rng, rng_step = jrd.split(rng, 2)
        action = jnp.argmax(model.apply(model_parameters, obs))
        n_obs, n_state, reward, done, _ = env.step(rng_step, state, action, env_params)

        new_val = n_obs, n_state, done, rng, cum_reward + reward
        return new_val

    val = lax.while_loop(
        lambda val: jnp.logical_not(val[2]), rollout_loop, (obs, state, False, rng, 0)
    )
    _, _, _, _, cum_reward = val

    return cum_reward


# ================================================
# ================   Brax   ======================
# ================================================


def get_braxv1_env(config: dict):
    from brax.v1.envs.wrappers import EpisodeWrapper as EpisodeWrapper_v1
    from brax.v1.envs import acrobot
    from brax.v1.envs import ant
    from brax.v1.envs import fast
    from brax.v1.envs import fetch
    from brax.v1.envs import grasp
    from brax.v1.envs import half_cheetah
    from brax.v1.envs import hopper
    from brax.v1.envs import humanoid
    from brax.v1.envs import humanoid_standup
    from brax.v1.envs import inverted_double_pendulum
    from brax.v1.envs import inverted_pendulum
    from brax.v1.envs import pusher
    from brax.v1.envs import reacher
    from brax.v1.envs import reacherangle
    from brax.v1.envs import swimmer
    from brax.v1.envs import ur5e
    from brax.v1.envs import walker2d

    _envs_v1 = {
        "acrobot": acrobot.Acrobot,
        "ant": ant.Ant,
        "fast": fast.Fast,
        "fetch": fetch.Fetch,
        "grasp": grasp.Grasp,
        "halfcheetah": half_cheetah.Halfcheetah,
        "hopper": hopper.Hopper,
        "humanoid": humanoid.Humanoid,
        "humanoidstandup": humanoid_standup.HumanoidStandup,
        "inverted_pendulum": inverted_pendulum.InvertedPendulum,
        "inverted_double_pendulum": inverted_double_pendulum.InvertedDoublePendulum,
        "pusher": pusher.Pusher,
        "reacher": reacher.Reacher,
        "reacherangle": reacherangle.ReacherAngle,
        "swimmer": swimmer.Swimmer,
        "ur5e": ur5e.Ur5e,
        "walker2d": walker2d.Walker2d,
    }
    return EpisodeWrapper_v1(
        _envs_v1[config["task"]["environnment"]](),
        episode_length=config["task"]["episode_length"],
        action_repeat=1,
    )


def get_braxv2_env(config: dict):
    from brax import envs as envs_v2
    from brax.envs.wrapper import EpisodeWrapper as EpisodeWrapper_v2

    env = envs_v2.get_environment(env_name=config["task"]["environnment"])
    return EpisodeWrapper_v2(
        env, episode_length=config["task"]["episode_length"], action_repeat=1
    )


def rollout_brax_task(
    config: dict,
    model: nn.Module,
    model_parameters: nn.FrozenDict,
    env,
    rng_reset: jrd.KeyArray,
) -> float:
    """Perform a single rollout of the environnment
    with the given policy parametrized by a `model` and its `model_parameters`.
    The return (sum of all the undiscounted rewards) is used as a fitness value.

    Args:
        config (dict): config of the current run.
        model (nn.Module): model used as a policy `f: S -> A`
        model_parameters (nn.FrozenDict): parameters of the model
        env (_type_): environnment used to perform the evaluation
        rng_reset (jrd.KeyArray): rng key used for evaluation

    Returns:
        float: return used as a fitness value.
    """
    state = jit(env.reset)(rng_reset)

    def rollout_loop(carry, x):
        env_state, cumulative_reward, active_episode = carry
        actions = model.apply(model_parameters, env_state.obs)
        new_state = jit(env.step)(env_state, actions)

        corrected_reward = new_state.reward * active_episode
        cumulative_reward = cumulative_reward + corrected_reward
        # NOTE - we carry over the active_episode value.
        # If it becomes 0, it stays at 0 because of the multiplication
        # this desactivated the reward counting in cumulative_reward by setting
        # the corrected_reward to 0
        new_active_episode = active_episode * (1 - new_state.done)
        return (new_state, cumulative_reward, new_active_episode), None

    active_episode = jnp.ones_like(state.reward)
    (_last_state, final_return, _), _ = lax.scan(
        f=rollout_loop,
        init=(state, state.reward, active_episode),
        xs=None,
        length=config["task"]["episode_length"],
    )

    return final_return


# ================================================
# ==============   Looks Good Prop   =============
# ================================================


def network_properties(model, model_parameters):
    """Takes a model and computes its properties"""
    f_expr = expressivity_ratio(model_parameters)

    mean, std = weights_distribution(model_parameters)
    f_w_distr = -jnp.abs(mean) - jnp.square(std - 0.5)

    (_, _), (out_mu, _) = input_distribution_restoration(model, model_parameters)
    f_inp = -jnp.abs(out_mu)

    return f_expr, f_w_distr, f_inp


def _network_properties_of_df(
    distance_function, rng_eval: jrd.KeyArray, meta_config: dict
):
    # First config of the curriculum tasks
    task_config = next(iter(meta_config["curriculum"].values()))
    assert "net" in task_config
    assert "layer_dimensions" in task_config["net"]
    assert "architecture" in task_config["net"]
    assert "encoding" in task_config
    assert "d" in task_config["encoding"]

    # NOTE - 1. Use DF to form a GENEDecoder
    decoder = GENEDecoder(task_config, distance_function)

    # NOTE - 2. Gen policy_genome
    dummy_network_genome = jrd.uniform(
        rng_eval, shape=(decoder.encoding_size(),), minval=-1.0, maxval=1.0
    )
    # NOTE - 3. Use GENEDecoder to decode the dummy policy genome into model params
    model = get_model(task_config)
    model_parameters = decoder.decode(dummy_network_genome)
    # NOTE - 4. Evaluate the policy networks parameters properties
    f_expr, f_w_distr, f_inp = network_properties(model, model_parameters)

    return f_expr, f_w_distr, f_inp


def evaluate_network_properties_nn_dist(
    distance_genome, rng_eval: jrd.KeyArray, meta_config: dict
):
    """This function evaluates the properties of a single policy network.
    The genome is the genome of the distance function
    """
    # NOTE - Use the genome to form a DF
    cgp_df = NNDistance(
        distance_genome, meta_config, meta_config["net"]["layer_dimensions"]
    )
    return _network_properties_of_df(cgp_df, rng_eval, meta_config)


def evaluate_network_properties_cgp_dist(
    cgp_genome, rng_eval: jrd.KeyArray, meta_config: dict
):
    """This function evaluates the properties of a single policy network.
    The genome is the genome of the distance function
    """
    # NOTE - Use cgp_genome to form a DF
    cgp_df = CGPDistance(cgp_genome, meta_config["cgp_config"])
    return _network_properties_of_df(cgp_df, rng_eval, meta_config)


def evaluate_rand_network_properties_n_times(
    df_genome,
    rng_eval: jrd.KeyArray,
    meta_config: dict,
    df_type: str = "nn",
    n: int = 32,
):
    if df_type == "nn":
        eval_f = vmap(
            partial(
                evaluate_network_properties_nn_dist,
                meta_config=meta_config,
            ),
            in_axes=(None, 0),
        )
    elif df_type == "cgp":
        eval_f = vmap(
            partial(
                evaluate_network_properties_cgp_dist,
                meta_config=meta_config,
            ),
            in_axes=(None, 0),
        )
    else:
        raise NotImplementedError

    keys = jrd.split(rng_eval, n)
    f_expr, f_w_distr, f_inp = eval_f(df_genome, keys)

    return (
        jnp.mean(f_expr),
        jnp.mean(f_w_distr),
        jnp.mean(f_inp),
    )


def eval_model_prop(policy_genotype, decoder, model):
    """
    1. Takes a policy genome, a decoder and a neural net model
    2. Evaluate LG prop of the policy neural net
    """
    model_parameters = decoder.decode(policy_genotype)
    return network_properties(model, model_parameters)
