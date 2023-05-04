from jax import jit, vmap, lax, default_backend
import jax.numpy as jnp
import jax.random as jrd
import jax
import chex
from brax import envs
from brax.envs.wrapper import EpisodeWrapper, VmapWrapper
from tqdm import tqdm
import evosax
import matplotlib.pyplot as plt

from functools import partial
from time import time

from gene.encoding import genome_to_model, gene_enc_genome_size
from gene.network import LinearModel


# FIXME: create network factory, rollout_loop out of the rollout, wolfwgnag banzhaf
def model_factory():
    # TODO: return / create a model with the right sizes, parially apply later on with config file
    return LinearModel([256, 6]).apply


def rollout(
    config: dict, model=None, _model_parameters=None, env=None, rng_reset=None
) -> float:
    state = jit(env.reset)(rng_reset)

    def rollout_loop(carry, x):
        env_state, model_parameters, cum_reward = carry
        # FIXME: problem seems to be here
        # actions = model.apply(model_parameters, env_state.obs)
        actions = model_factory()(model_parameters, env_state.obs)
        new_state = jit(env.step)(env_state, actions)

        corrected_reward = new_state.reward * (1 - new_state.done)
        new_carry = new_state, model_parameters, cum_reward + corrected_reward
        # NOTE: New_state or env_state?
        return new_carry, corrected_reward

    carry, returns = lax.scan(
        f=rollout_loop,
        init=(state, _model_parameters, state.reward),
        xs=None,
        length=config["problem"]["episode_length"],
    )

    jax.debug.print(
        "[Debug]: {carry} | rewards={rewards}", carry=carry[-1], rewards=returns[:5]
    )

    chex.assert_trees_all_close(carry[-1], jnp.cumsum(returns)[-1])

    """
    [Debug]: return = -218051395977216.0 | rewards = [-6.4879230e+04 -2.1647006e+10 -9.7899263e+11 -1.2201745e+12 -1.7165093e+10]
    [Debug]: return = -266524799533056.0 | rewards = [-4.0530293e+04 -9.2708517e+11 -2.9265510e+12 -4.8063349e+10 -2.6935624e+11]
    [Debug]: return = -167855559540736.0 | rewards = [-5.5108206e+05 -7.3815307e+11 -4.4827515e+11 -4.6923661e+11 -1.7507236e+11]
    [Debug]: return = -168385350467584.0 | rewards = [-2.1220669e+05 -1.5440354e+11 -1.7505509e+11 -4.0159056e+10 -4.2847306e+10]
    [Debug]: return = -226016949698560.0 | rewards = [-5.4868881e+05 -8.9942740e+11 -6.9246773e+11 -3.7358830e+11 -4.5857024e+11]
    """

    return carry[1]


def evaluate_individual(
    genome: jnp.array,
    rng: jrd.KeyArray,
    config: dict,
    env,
) -> float:
    model, model_parameters = genome_to_model(genome, config=config)

    fitness = rollout(
        model=model,
        _model_parameters=model_parameters,
        config=config,
        env=env,
        rng_reset=rng,
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
    env = EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )
    # env = VmapWrapper(env, batch_size=config["evo"]["population_size"])

    vmap_evaluate_individual = vmap(
        partial(evaluate_individual, config=config, env=env)
    )
    jit_vmap_evaluate_individual = jit(vmap_evaluate_individual)

    for _generation in tqdm(range(config["evo"]["n_generations"])):
        # RNG key creation for downstream usage
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)
        # NOTE - Evaluate
        # temp_fitness = jit_vmap_evaluate_individual(x, rng_eval)
        temp_fitness = jnp.array(
            [evaluate_individual(genome, rng_eval, config, env) for genome in x]
        )
        fitness = -1 * temp_fitness

        print(temp_fitness[:4], fitness[:4])

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        # Log / stats step: Add the fitness to log object
        log = es_logging.update(log, x, temp_fitness)

    return state, es_logging, log


config = {
    "evo": {"strategy_name": "xNES", "n_generations": 200, "population_size": 100},
    "net": {"layer_dimensions": [17, 256, 6]},
    "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
    "problem": {"environnment": "halfcheetah", "maximize": True, "episode_length": 400},
}

if __name__ == "__main__":
    assert default_backend() == "gpu"

    state, es_logging, log = run(config)
    es_logging.plot(log, "Brax half cheetah learning 2")
    plt.savefig(f"{str(time())}_brax_v2.png")
