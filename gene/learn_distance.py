"""
Goal: learn a distance function that maximizes the Expected fitness value of each genome

- max_d [E_(x in D_gene) fit(d(x))]

D_gene variable aléatoire qui renvoie un genome compressé GENE


- Use direct encoding for distance function network
- Optimize distance function network

1. init all and define func (simple net)
2. Learn loop:
    1. Ask for candidate distance functions
    2. Evaluate these distance function (in parallel)
    3. Tell the fitness and update state
3. Return last best indiv
"""
from time import time
from functools import partial

import jax.random as jrd
import jax.numpy as jnp
from jax import Array, jit, vmap, tree_util
import flax.linen as nn
import evosax
import chex

from gene.encoding import (
    _direct_enc_genome_size,
    gene_decoding_w_dist,
    gene_enc_genome_size,
)
from gene.evaluate import get_brax_env, _rollout_brax
from gene.tracker import batch_wandb_log
from gene.encoding import _direct_decoding
from gene.network import LinearModel, BoundedLinearModel


# ========================================
# =============== Helpers ================
# ========================================
def dict_of_arrays_to_array_of_flat_phenotype(tree, config: dict):
    """Transforms a dict of parameters of `n` networks (phenotypes) to
    a flattened representation of `n` penotypes.

    {[20, n]} to [20, {n}].
    """
    flat_tree = tree_util.tree_flatten(tree)[0]

    trans_tree = jnp.array(
        [
            jnp.concatenate(tree_util.tree_map(lambda e: jnp.ravel(e[i]), flat_tree))
            for i in range(flat_tree[0].shape[0])
        ]
    )
    return trans_tree


@jit
def genome_distance_from_center(genomes: Array, center: Array) -> float:
    """Compute the mean distance between the genomes and the center

    Args:
        genomes (Array): The genomes on which to evaluate the distance.
        center (Array): The center of the genomes

    Returns:
        float: Distance.
    """
    chex.assert_equal(
        genomes.shape[1],
        center.shape[0],
    )
    distances = jnp.linalg.norm(genomes - center, ord=2, axis=1)
    # distances should be a matrix with leading axis shape 20 (config)
    return jnp.mean(distances, axis=0)


# ========================================
# ============ End helpers ===============
# ========================================


@chex.dataclass
class NNDistance:
    """Neural Network Distance function.

    This distance function uses a neural network to compute
    the distance between two vectors.
    """

    distance_genome: Array
    layer_dimensions: tuple[int]

    def __post_init__(self):
        self._genome_to_nn_distance_model(self.distance_genome, self.layer_dimensions)

    # @partial(jit, static_argnums=(0))
    def _apply(self, x) -> float:
        return self.model.apply(self.model_parameters, x)

    def nn_distance(self, x, n1_i, n2_i):
        return self._apply(jnp.concatenate((x[n1_i], x[n2_i])))

    @property
    def vmap_evaluate(self):
        """Returns the vectorized version of `nn_distance` and caches it.
        Can then be used to retrieve the vectorized function"""
        # Create vectorized function if not already created
        if not hasattr(self, "_f"):
            # FIXME - strange behaviour, adds an axis
            self._f = jit(
                vmap(
                    vmap(self.nn_distance, in_axes=(None, None, 0)),
                    in_axes=(None, 0, None),
                )
            )
        return self._f

    def _genome_to_nn_distance_model(
        self, distance_genome: jnp.ndarray, layer_dimensions: list[int]
    ):
        self.model_parameters: nn.Module = nn.FrozenDict(
            {"params": _direct_decoding(distance_genome, layer_dimensions)}
        )
        self.model: nn.FrozenDict = LinearModel(layer_dimensions[1:])


def evaluate_individual_brax_w_distance(
    genome: jnp.array, rng: jrd.KeyArray, config: dict, env, distance: NNDistance
) -> float:
    model = BoundedLinearModel(config["net"]["layer_dimensions"][1:])
    model_parameters = nn.FrozenDict(
        {
            "params": gene_decoding_w_dist(
                genome, config=config, distance_network=distance
            )
        }
    )

    # print(tree_util.tree_map(lambda x: x.shape, model_parameters))
    fitness = _rollout_brax(
        model=model,
        model_parameters=model_parameters,
        config=config,
        env=env,
        rng_reset=rng,
    )
    return fitness, model_parameters


def evaluate_distance_f(
    distance_genome: Array,
    rng_sample: jrd.KeyArray,
    rng_eval: jrd.KeyArray,
    gene_sample_size: int,
    config: dict,
    distance_layer_dimensions: tuple[int],
):
    """Evaluate a single `distance_genome` function.

    Args:
        distance_genome (Array): the genome of the distance function.
        config (dict): config dict of the current run.
        rng (jrd.KeyArray): rng key to evaluate the distance function.

    Returns:
        float: fitness of the mean of the evaluated sampled population
            using the parametrized distance function.
    """
    # 1. Sample GENE individuals
    # TODO - Use a smaller sigma (0.5, 0.1) for gene sampling
    # "variance" of the normal distribution
    sigma = config["distance_network"]["sample_sigma"]
    # TODO - Try with high sigma value ?
    sampled_gene_individuals_genomes = (
        jrd.normal(
            rng_sample,
            shape=(
                gene_sample_size,
                gene_enc_genome_size(config),
            ),
        )
        * sigma
    )
    # 2. Generate the parametrized distance fun (the model and the model parameters)
    distance = NNDistance(
        distance_genome=distance_genome, layer_dimensions=distance_layer_dimensions
    )
    # 3. Use distance func to evaluate all sampled individuals
    env = get_brax_env(config)

    _partial_evaluate_individual = partial(
        evaluate_individual_brax_w_distance,
        rng=rng_eval,
        config=config,
        env=env,
        distance=distance,
    )
    jit_vmap_evaluate_individual = jit(vmap(_partial_evaluate_individual, in_axes=(0,)))

    fitnesses, all_models_parameters = jit_vmap_evaluate_individual(
        sampled_gene_individuals_genomes
    )

    # We need to extract the flattened representation of the networks parameters
    # to compute statistics
    flat_model_parameters = dict_of_arrays_to_array_of_flat_phenotype(
        all_models_parameters, config
    )

    # 3. Average all fitnesses and return the value
    # TODO - penalize fitness based on the variance (if too high, reduce fitness)
    _penality = 1.0
    # NOTE - Log all stats (mean, median, variance, ..stddev)
    statistics = {
        "fitness": {
            "mean": jnp.mean(fitnesses, axis=0),
            "median": jnp.median(fitnesses),
            "variance": jnp.var(fitnesses),  # stddev can be derived from var
            "min": jnp.min(fitnesses),
            "max": jnp.max(fitnesses),
        },
        # neural network parameters stats
        "phenotypes": {
            # distance from the measured center of the phenotypes
            "dist_from_emp_center": genome_distance_from_center(
                genomes=flat_model_parameters,
                center=jnp.mean(flat_model_parameters, axis=0),
            ),
            # distance from the projected center of the phenotypes
            "dist_from_center": genome_distance_from_center(
                genomes=flat_model_parameters,
                center=jnp.zeros_like(flat_model_parameters[0]),
            ),
        },
        "genotypes": {
            "dist_from_emp_center": genome_distance_from_center(
                genomes=sampled_gene_individuals_genomes,
                center=jnp.mean(sampled_gene_individuals_genomes, axis=0),
            ),
            "dist_from_center": genome_distance_from_center(
                genomes=sampled_gene_individuals_genomes,
                center=jnp.zeros_like(sampled_gene_individuals_genomes[0]),
            ),
        },
    }
    return statistics


def learn_distance_f_evo(config: dict, wdb_run):
    """Learn a distance function that maximizes the fitness
    of the evaluated gene-encoded networks.

    f(d(x))

    Args:
        config (dict): config of the run
    """
    distance_layer_dimensions = config["distance_network"]["layer_dimensions"]
    dist_f_n_dimensions: int = _direct_enc_genome_size(distance_layer_dimensions)

    rng = jrd.PRNGKey(config["seed"])
    rng, rng_init = jrd.split(rng, 2)

    strategy = evosax.Strategies[config["distance_network"]["evo"]["strategy_name"]](
        popsize=config["distance_network"]["evo"]["population_size"],
        num_dims=dist_f_n_dimensions,
    )

    es_params = strategy.default_params.replace(init_min=-2, init_max=2)
    state = strategy.initialize(rng_init, es_params)

    partial_evaluate_distance_f = partial(
        evaluate_distance_f,
        config=config,
        distance_layer_dimensions=distance_layer_dimensions,
    )
    vectorized_evaluate_distance_f = jit(
        vmap(partial_evaluate_distance_f, in_axes=(0, 0, None, None)),
        static_argnames=["gene_sample_size"],
    )

    for _generation in range(config["distance_network"]["evo"]["n_generations"]):
        print(f"[Log] - gen {_generation} @ {time()}")
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # + 1 to create a new rng and +1 for evaluating the population mean
        (
            rng,
            rng_sample_center,
            *rng_sample,
        ) = jrd.split(rng, config["distance_network"]["evo"]["population_size"] + 2)
        rng_sample = jnp.array(rng_sample)
        # NOTE - Ask
        x, state = strategy.ask(rng_gen, state, es_params)

        # NOTE - Evaluate
        # statistics -> array of elements, one for each distance_individual
        statistics = vectorized_evaluate_distance_f(
            x,
            rng_sample,
            rng_eval,
            config["distance_network"]["gene_sample_size"],
        )
        fitness = (
            -1 * statistics["fitness"]["mean"]
        )  # we want to maximize the objective f.
        print(statistics, "\n")

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = strategy.tell(x, fitness, state, es_params)

        center_stats = partial_evaluate_distance_f(
            state.mean,
            rng_sample_center,
            rng_eval,
            config["distance_network"]["gene_sample_size"] * 10,
        )
        # TODO - log to W&B:
        # - sample mean fitness
        # - empirical variance
        # - empirical mean
        # NOTE - Log
        batch_wandb_log(
            wdb_run,
            statistics,
            config["distance_network"]["gene_sample_size"],
            prefix="dist_individual",
        )

    # returns fitnesses
    return statistics["fitness"]["mean"], center_stats
