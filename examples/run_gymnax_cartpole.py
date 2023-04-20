import jax.numpy as jnp
import jax.random as jrd
import jax
import gymnax
from gymnax.visualize import Visualizer
from jax_vmap import genome_to_param
from evosax import FitnessShaper, OpenES
from tqdm import tqdm
from evosax.utils import ESLog
import matplotlib.pyplot as plt

import time
from functools import partial


class Modulax:
    def __init__(self) -> None:
        pass

    def __call__(self, *args) -> jnp.ndarray:
        return self.forward(*args)

    def forward(self, x):
        raise NotImplementedError


class ReLU(Modulax):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return jnp.maximum(x, 0)


class Sigmoid(Modulax):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return 1 / (1 + jnp.exp(-x))


class Linear(Modulax):
    def __init__(
        self,
        in_features: jnp.ndarray,
        out_features: jnp.ndarray,
        parameters: dict = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        if parameters is None:
            self.parameters = {
                "w": None,
                # 'b': None,
            }
        else:
            self.parameters = parameters

    def forward(self, x):
        assert self.parameters["w"].shape == (self.in_features, self.out_features)
        # assert self.parameters['b'].shape == (self.out_features, )
        return x @ self.parameters["w"]  #  + self.parameters['b']


class Sequential(Modulax):
    def __init__(self, *args) -> None:
        self.layers: list[Linear] = args

    def __call__(self, *args) -> jnp.ndarray:
        return self.forward(*args)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def evaluate_cartpole(
    genome: jnp.ndarray, layer_dimensions: list, rng_eval, save: bool = False
):
    # SECTION genome to model
    model_parameters = genome_to_param(genome, d=1, layer_dimensions=layer_dimensions)
    model = Sequential(
        Linear(
            in_features=layer_dimensions[0],
            out_features=layer_dimensions[1],
            parameters=model_parameters[0],
        ),
        ReLU(),
        Linear(
            in_features=layer_dimensions[1],
            out_features=layer_dimensions[2],
            parameters=model_parameters[1],
        ),
        Sigmoid(),
    )
    # !SECTION
    rng = rng_eval
    rng, rng_reset = jrd.split(rng, 2)

    env, env_params = gymnax.make("CartPole-v1")
    obs, state = env.reset(rng_reset, env_params)

    state_seq, reward_seq = [], []
    done = False
    while not done:
        # NOTE: Stats
        state_seq.append(state)

        rng, rng_step = jrd.split(rng, 2)
        # Sample a random action.
        action_disrtibution = model(obs)
        action = jnp.argmax(action_disrtibution)

        # Perform the step transition.
        n_obs, n_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        # NOTE: Stats
        reward_seq.append(reward)

        obs = n_obs
        state = n_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    if save:
        vis = Visualizer(env, env_params, state_seq, cum_rewards)
        vis.animate(
            f"anim/{int(time.time())}__cartpole_r_{cum_rewards[-1]}.gif", view=False
        )
    return cum_rewards[-1]


def run():
    LAYER_DIMS = [4, 32, 2]
    MAX_GEN = 20
    POP_SIZE = 20
    N_NEURONS = sum(LAYER_DIMS)

    fit_shaper = FitnessShaper(maximize=True)
    rng = jrd.PRNGKey(1)
    strategy = OpenES(popsize=POP_SIZE, num_dims=N_NEURONS)
    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng, es_params)

    es_logging = ESLog(
        num_dims=N_NEURONS, num_generations=MAX_GEN, top_k=3, maximize=True
    )
    log = es_logging.initialize()

    vmap_evaluate_cartpole = jax.vmap(
        partial(evaluate_cartpole, layer_dimensions=LAYER_DIMS, save=True)
    )

    for generation in tqdm(range(MAX_GEN), desc="Generation loop", position=0):
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        x, state = strategy.ask(rng_gen, state, es_params)
        # Evaluate generation[s] genome
        # TODO: vmap but partial applications is necessary
        # temp_fitness = jnp.array([evaluate_cartpole(genome, LAYER_DIMS, rng_eval) for genome in x])
        temp_fitness = vmap_evaluate_cartpole(x, rng_eval)
        print(temp_fitness)

        fitness = fit_shaper.apply(x, temp_fitness)
        # Tell / Update step
        state = strategy.tell(x, fitness, state, es_params)

        # Update the log with results
        log = es_logging.update(log, x, fitness)

        if (generation + 1) % 5 == 0:
            print(
                f'[INFO - OpenES]: Gen: {generation+1} | Fitness {log["log_top_1"][generation]}'
            )  # state.best_fitness:.5f}

    print(state.best_fitness)
    fig, ax = es_logging.plot(
        log,
        "CartPole Env, Gene encoding",
    )  # ylims=(0, 30)
    plt.savefig(fname=f"figures/{int(time.time())}_cartpole_perf")
    # take best and save
    evaluate_cartpole(state.best_member, LAYER_DIMS, jrd.split(rng)[1], save=True)
    return state.best_member


if __name__ == "__main__":
    best = run()
    print(best)


# FIXME: Should be instant if vmap-ed and jit-ed
