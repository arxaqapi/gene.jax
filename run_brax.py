import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap, default_backend
import evosax
from brax import envs

from gene.encoding import Encoding_size_function
from gene.evaluate import genome_to_model

from functools import partial


config = {
    "evo": {"strategy_name": "SNES", "n_generations": 100, "population_size": 20},
    "net": {"layer_dimensions": [18, 64, 6]},
    "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
    "problem": {"environnment": "halfcheetah", "maximize": True},
}


def rollout(config: dict, model=None, model_parameters=None) -> float:
    # https://github.com/google/brax/blob/main/brax/envs/half_cheetah.py
    # action space values in [-1 to 1]
    env_name = config["problem"]["environnment"]
    env = envs.get_environment(env_name)
    # state = jit(env.reset)(rng=jrd.PRNGKey(seed=0))
    state = env.reset(jrd.PRNGKey(0))

    rewards = []
    # FIXME: prob with boolean
    while not state.done:
        action = jnp.argmax(model.apply(model_parameters, state.obs))
        print(action)
        print(state.obs)
        print(state.done)
        exit(15)
        # FIXME: error (ValueError: axis 0 is out of bounds for array of dimension 0)
        state = env.step(state, action)
        rewards.append(state.reward)
    
    cum_rewards = jnp.cumsum(rewards)
    return cum_rewards[-1]


def evaluate_individual(
    genome: jnp.array,
    config: dict,
) -> float:
    # genome: 334
    model, model_parameters = genome_to_model(genome, config=config)

    fitness = rollout(
        model=model,
        model_parameters=model_parameters,
        config=config,
    )
    return fitness


def run(config: dict, rng: jrd.KeyArray = jrd.PRNGKey(5)):
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

    for generation in range(config["evo"]["n_generations"]):
        # RNG key creation for downstream usage
        rng, rng_gen = jrd.split(rng, 2)
        # Here, each individual has an unique random key used for evaluation purposes
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        temp_fitness = [evaluate_individual(genome, config) for genome in x]
        # temp_fitness = jit_vmap_evaluate_individual(x)
        fitness = fit_shaper.apply(x, temp_fitness)

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # Log / stats step: Add the fitness to log object
        log = es_logging.update(log, x, temp_fitness)

    return state, log


if __name__ == "__main__":
    state, log = run(config)
    print(log["top_fitness"])
