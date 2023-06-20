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
from pathlib import Path
import pickle
import wandb

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

    def save_parameters(self, path: Path) -> None:
        """Saves the `model_parameters` to `path`.

        Args:
            path (Path): The Path and name of the file where it will be saved
        """
        with path.open("wb") as f:
            pickle.dump(self.model_parameters, f)

    def load_parameters(self, path: Path) -> None:
        """Load the saved `model_parameters` from `path` to `self.model_parameters`.

        Args:
            path (Path): The Path and name of the file to retrieve.
        """
        with path.open("rb") as f:
            self.model_parameters = pickle.load(f)


def evaluate_individual_brax_w_distance(
    genome: jnp.array, rng: jrd.KeyArray, config: dict, env, distance: NNDistance
) -> tuple[float, nn.FrozenDict]:
    """Evaluates a single individual `genome` using a parametrized distance function `distance`.


    Args:
        genome (jnp.array): _description_
        rng (jrd.KeyArray): _description_
        config (dict): _description_
        env (_type_): _description_
        distance (NNDistance): _description_

    Returns:
        tuple[float, nn.FrozenDict]: the fitness as a float
            and the `model_parameters` of the decoded genome
    """
    model = BoundedLinearModel(config["net"]["layer_dimensions"][1:])
    model_parameters: nn.FrozenDict = nn.FrozenDict(
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


def _sample_arround_genome(
    rng_sample: jrd.KeyArray,
    gene_sample_size: int,
    config: dict,
    sample_center_genome: Array = None,
):
    """Sample gene genomes arround `sample_center_genome` or the point `(0,...,0)`.
    The genomes are sampled using a normal distribution with parameter `sigma`,
    found in the `config` file.

    Args:
        rng_sample (jrd.KeyArray): rng key used to sample arround
            the `sample_center_genome` point.
        gene_sample_size (int): Amount of samples drawn.
        config (dict): config dict of the current run.
        sample_center_genome (Array, optional): _description_. Defaults to None.

    Returns:
        Array: (gene_sample_size, D)
    """
    sigma = config["distance_network"]["sample_sigma"]
    noise = (
        jrd.normal(
            rng_sample,
            shape=(
                gene_sample_size,
                gene_enc_genome_size(config),
            ),
        )
        * sigma
    )
    return sample_center_genome + noise


def evaluate_distance_f(
    distance_genome: Array,
    rng_sample: jrd.KeyArray,
    rng_eval: jrd.KeyArray,
    gene_sample_size: int,
    config: dict,
    distance_layer_dimensions: tuple[int],
    sample_center_genome: Array,
):
    """Evaluate a single `distance_genome` function.

    Args:
        distance_genome (Array): the genome of the distance function.
        rng_sample (jrd.KeyArray): rng key used to sample arround the gene genomes
        rng_eval (jrd.KeyArray): rng key to evaluate the distance function.
        gene_sample_size (int): how many samples are drawn and evaluated
            for the fitness value.
        config (dict): config dict of the current run.
        distance_layer_dimensions (tuple[int]): the layer dimensions of the distance
            function neural network.

    Returns:
        float: fitness of the mean of the evaluated sampled population
            using the parametrized distance function.
    """
    chex.assert_tree_no_nones(sample_center_genome)
    # 1. Sample GENE individuals
    # mu: sample_center_genome | sigma: config[""]["sigma"]
    sampled_gene_individuals_genomes = _sample_arround_genome(
        rng_sample, gene_sample_size, config, sample_center_genome
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

    # NOTE - all_models_parameters are the penotypes
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
    _penality = 1.0 * (jnp.max(fitnesses) - jnp.min(fitnesses))
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
        # SECTION - to fix
        # "phenotypes": {
        #     # distance from the measured center of the phenotypes
        #     "dist_from_emp_center": genome_distance_from_center(
        #         genomes=flat_model_parameters,
        #         center=jnp.mean(flat_model_parameters, axis=0),
        #     ),
        #     # distance from the projected center of the phenotypes
        #     "dist_from_center": genome_distance_from_center(
        #         genomes=flat_model_parameters,
        #         # NOTE - center is the projected GENE genome into the phenotype space
        #         # FIXME - Project genome_to_model to get the correct center
        #         center=jnp.zeros_like(flat_model_parameters[0]),
        #     ),
        # },
        # "genotypes": {
        #     "dist_from_emp_center": genome_distance_from_center(
        #         genomes=sampled_gene_individuals_genomes,
        #         center=jnp.mean(sampled_gene_individuals_genomes, axis=0),
        #     ),
        #     "dist_from_center": genome_distance_from_center(
        #         genomes=sampled_gene_individuals_genomes,
        #         center=sample_center_genome,
        #     ),
        # },
        # !SECTION - to fix
    }
    return statistics


def learn_distance_f_evo(config: dict, wdb_run, sample_center_genome: Array):
    """Learn a distance function that maximizes the fitness
    of the evaluated gene-encoded networks.

    f(d(x))

    Args:
        config (dict): config of the run
    """
    assert wdb_run is not None

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
        sample_center_genome=sample_center_genome,
    )
    vectorized_evaluate_distance_f = jit(
        vmap(partial_evaluate_distance_f, in_axes=(0, 0, None, None)),
        static_argnames=["gene_sample_size"],
    )

    for _generation in range(config["distance_network"]["evo"]["n_generations"]):
        print(f"[Log] - gen {_generation} @ {time()}")
        rng, rng_gen, rng_eval = jrd.split(rng, 3)
        # + 1 to create a new rng
        # + 1 for evaluating the population mean
        # + 1 for evaluating the best member
        (
            rng,
            rng_sample_center,
            rng_best_member,
            *rng_sample,
        ) = jrd.split(rng, config["distance_network"]["evo"]["population_size"] + 3)
        rng_sample = jnp.array(rng_sample)

        # NOTE - Ask
        x, state = jit(strategy.ask)(rng_gen, state, es_params)

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

        # NOTE - Tell: overwrites current strategy state with the new updated one
        state = jit(strategy.tell)(x, fitness, state, es_params)

        # Evaluate current center of the population
        center_stats = partial_evaluate_distance_f(
            state.mean,
            rng_sample_center,
            rng_eval,
            config["distance_network"]["gene_sample_size"] * 25,
        )
        # Evaluate current best individual
        best_member_stats = partial_evaluate_distance_f(
            state.best_member,
            rng_best_member,
            rng_eval,
            config["distance_network"]["gene_sample_size"] * 25,
        )
        # NOTE - Log stats, center_stats and best_member_stats to w&b
        # FIXME - merge log events
        batch_wandb_log(
            wdb_run,
            statistics,
            config["distance_network"]["gene_sample_size"],
            prefix="dist_individual",
        )
        wdb_run.log({"center_stats": center_stats})
        wdb_run.log({"best_member_stats": best_member_stats})

    best_distance_f = NNDistance(
        distance_genome=state.best_member, layer_dimensions=distance_layer_dimensions
    )
    save_path = Path("best_member")
    # save best_member_params to file and w&b as artifact
    best_distance_f.save_parameters(save_path)

    artifact = wandb.Artifact(name="best_member_model_parameters", type="model")
    artifact.add_file(
        local_path=save_path,
    )
    wdb_run.log_artifact(artifact)

    return statistics, center_stats, best_member_stats