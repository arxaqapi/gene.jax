import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap, default_backend, lax
import evosax
from brax import envs
from brax.envs.wrappers import EpisodeWrapper
from tqdm import tqdm
import matplotlib.pyplot as plt

from gene.encoding import Encoding_size_function
from gene.evaluate import genome_to_model

from functools import partial
from time import time


config = {
    "evo": {"strategy_name": "SNES", "n_generations": 50, "population_size": 35},
    "net": {"layer_dimensions": [18, 64, 6]},
    "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
    "problem": {"environnment": "halfcheetah", "maximize": True},
}


def _rollout_problem_lax(config: dict, model=None, model_parameters=None, rng=None) -> float:
    env = envs.get_environment(config["problem"]["environnment"])
    env = EpisodeWrapper(env, episode_length=500, action_repeat=1)
    state = jit(env.reset)(rng)

    def rollout_loop(val):
        state, cum_reward = val

        actions = model.apply(model_parameters, state.obs)
        state = jit(env.step)(state, actions)

        new_val = state, cum_reward + state.reward
        return new_val

    val = lax.while_loop(
        lambda val: jnp.logical_not(val[0].done), rollout_loop, (state, 0)
    )
    _, cum_reward = val

    return cum_reward


def _rollout_problem_scan(config: dict, model=None, model_parameters=None, rng=None, episode_length=500) -> float:
    # https://github.com/google/brax/blob/c2cd14cf762242d63aeec106d955390c8e14d582/brax/training/agents/es/train.py#L144
    env = envs.get_environment(config["problem"]["environnment"])
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
    state = jit(env.reset)(rng)

    def rollout_loop(carry, x):
        env_state  = carry

        actions = model.apply(model_parameters, env_state.obs)
        new_state = jit(env.step)(env_state, actions)

        new_carry = new_state
        # New_state or env_state?
        return new_carry, new_state.reward * (1 - new_state.done)

    carry, rewards = lax.scan(
        f=rollout_loop,
        init=state,
        xs=None,
        length=episode_length)

    return jnp.cumsum(rewards)[-1]



def evaluate_individual(
    genome: jnp.array,
    rng: jrd.KeyArray,
    config: dict,
) -> float:
    model, model_parameters = genome_to_model(genome, config=config)

    fitness = _rollout_problem_lax(
        model=model,
        model_parameters=model_parameters,
        config=config,
        rng=rng
    )
    return fitness


def run(config: dict, rng: jrd.KeyArray = jrd.PRNGKey(0)):
    num_dims = Encoding_size_function[config["encoding"]["type"]](config)

    rng, rng_init = jrd.split(rng, 2)
    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=num_dims,
    )

    fit_shaper = evosax.FitnessShaper(maximize=config["problem"]["maximize"])
    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng_init, es_params)

    # Enable logging data during training process
    es_logging = evosax.ESLog(
        num_dims=num_dims,
        num_generations=config["evo"]["n_generations"],
        top_k=5,
        maximize=True,
    )
    log = es_logging.initialize()

    vmap_evaluate_individual = vmap(partial(evaluate_individual, config=config))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    for generation in tqdm(range(config["evo"]["n_generations"])):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        rng_eval_v = jrd.split(rng_eval, config["evo"]["population_size"])
        # Here, each individual has an unique random key used for evaluation purposes
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        # temp_fitness = jnp.array([evaluate_individual(genome, k, config) for genome, k in zip(x, rng_eval_v)])
        temp_fitness = jit_vmap_evaluate_individual(x, rng_eval_v)
        fitness = fit_shaper.apply(x, temp_fitness)

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # Log / stats step: Add the fitness to log object
        log = es_logging.update(log, x, temp_fitness)

    return state, es_logging, log


if __name__ == "__main__":
    assert default_backend() == "gpu"
    print("[Info] - Let's gong\n")

    state, es_logging, log = run(config)
    es_logging.plot(log, "Brax half cheetah learning")
    plt.savefig(f"{str(time())}_there.png")

    print("\n\n[Info] - Finished")
