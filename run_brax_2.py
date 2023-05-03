
from jax import jit, vmap, lax, default_backend
import jax.numpy as jnp
import jax.random as jrd
import jax
from brax import envs
from brax.envs.wrappers import EpisodeWrapper, VmapWrapper
from tqdm import tqdm
import evosax
import matplotlib.pyplot as plt

from functools import partial
from time import time

from gene.encoding import genome_to_model, gene_enc_genome_size


# def rollout(config: dict, model=None, model_parameters=None, env=None, rng_reset=None) -> float:
#     state = jit(env.reset)(rng_reset)

#     def rollout_loop(carry, x):
#         env_state  = carry

#         actions = model.apply(model_parameters, env_state.obs)
#         new_state = jit(env.step)(env_state, actions)

#         new_carry = new_state
#         # New_state or env_state?
#         return new_carry, new_state.reward * (1 - new_state.done)

#     carry, rewards = lax.scan(
#         f=rollout_loop,
#         init=state,
#         xs=None,
#         length=config["problem"]["episode_length"])

#     jax.debug.print(carry)

#     return jnp.cumsum(rewards)[-1]


def _rollout_problem_lax(config: dict, model=None, model_parameters=None, env=None, rng_reset=None) -> float:
    state = env.reset(rng_reset)
    # state = jit(env.reset)(rng_reset)

    def rollout_loop(val):
        state, cum_reward = val

        actions = model.apply(model_parameters, state.obs)
        new_state = env.step(state, actions)
        # state = jit(env.step)(state, actions)

        new_val = new_state, cum_reward + new_state.reward
        return new_val

    val = lax.while_loop(
        lambda val: jnp.logical_not(val[0].done), rollout_loop, (state, state.reward)
    )
    _, cum_reward = val

    return cum_reward



def evaluate_individual(
    genome: jnp.array,
    rng: jrd.KeyArray,
    config: dict,
    env,
) -> float:
    model, model_parameters = genome_to_model(genome, config=config)

    # fitness = rollout(
    fitness = _rollout_problem_lax(
        model=model,
        model_parameters=model_parameters,
        config=config,
        env=env,
        rng_reset=rng
    )
    return fitness


def run(config: dict, rng: jrd.KeyArray = jrd.PRNGKey(5)):
    num_dims = gene_enc_genome_size(config)

    rng, rng_init = jrd.split(rng, 2)
    strategy = evosax.Strategies[config["evo"]["strategy_name"]](
        popsize=config["evo"]["population_size"],
        num_dims=num_dims,
    )

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

    env = envs.get_environment(env_name=config["problem"]["environnment"])
    env = EpisodeWrapper(env, episode_length=config["problem"]["episode_length"],action_repeat=1)
    # env = VmapWrapper(env, batch_size=config["evo"]["population_size"])

    vmap_evaluate_individual = vmap(partial(evaluate_individual, config=config, env=env))
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    for generation in tqdm(range(config["evo"]["n_generations"])):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        rng_eval_v = jrd.split(rng_eval, config["evo"]["population_size"])
        # Here, each individual has an unique random key used for evaluation purposes
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        temp_fitness = jit_vmap_evaluate_individual(x, rng_eval_v)
        # temp_fitness = jnp.array([evaluate_individual(genome, config, env) for genome in x])
        fitness = -1 * temp_fitness

        print(temp_fitness)

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # Log / stats step: Add the fitness to log object
        log = es_logging.update(log, x, temp_fitness)

    return state, es_logging, log

config = {
    "evo": {"strategy_name": "xNES", "n_generations": 30, "population_size": 100},
    "net": {"layer_dimensions": [18, 256, 6]},
    "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
    "problem": {"environnment": "halfcheetah", "maximize": True, "episode_length": 400},
}

if __name__ == "__main__":
    assert default_backend() == "gpu"

    state, es_logging, log = run(config)
    es_logging.plot(log, "Brax half cheetah learning 2")
    plt.savefig(f"{str(time())}_brax_v2.png")