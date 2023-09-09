"""
1. Learn a neural network distance function using meta-evolution 
2. Extract the distance function and use it to train a policy
3. Compare the obtained policy with those obtained by training with:
    1. GENE w. pL2 encoding
    1. GENE w. L2 encoding
    1. Direct encoding
"""
import time
import wandb
from copy import deepcopy

from gene.utils import fail_if_not_device, load_config, validate_json
from gene.experiment import Experiment
from gene.meta import meta_learn_nn_corrected
from gene.core.distances import NNDistance


DEVNULL = "devnull"


if __name__ == "__main__":
    fail_if_not_device()

    expe_time = int(time.time())

    # NOTE - 1. Meta nn
    meta_nn_config = load_config("config/nn_meta_df_corrected.json")
    _, best_df_genome = meta_learn_nn_corrected(
        meta_nn_config,
        wandb.init(
            project=DEVNULL,
            name="CC-Comp-meta",
            config=meta_nn_config,
            tags=[expe_time],
        ),
        beta=1,
    )

    # NOTE - config setup
    base_config = deepcopy(meta_nn_config["curriculum"]["hc_500"])
    base_config["task"]["episode_length"] = 1000
    base_config["evo"]["population_size"] = 256

    # NOTE - 2. Use distance function to train a policy
    nn_df_config = deepcopy(base_config)
    nn_df_config["encoding"]["distance"] = ""
    nn_df_config["group"] = "learned"
    validate_json(nn_df_config)

    Experiment(
        nn_df_config,
        wandb.init(
            project=DEVNULL,
            name="CC-Comp-learned-nn",
            config=nn_df_config,
            tags=[expe_time],
        ),
        distance_function=NNDistance(
            distance_genome=best_df_genome,
            config={
                "net": {
                    "layer_dimensions": [6, 32, 32, 1],
                    "architecture": "tanh_linear",
                }
            },
        ),
    )

    # NOTE - 3.1. GENE w. pL2
    conf_gene_pl2 = deepcopy(base_config)
    conf_gene_pl2["encoding"]["distance"] = "pL2"
    conf_gene_pl2["encoding"]["type"] = "gene"
    conf_gene_pl2["group"] = "pL2"
    validate_json(conf_gene_pl2)

    Experiment(
        conf_gene_pl2,
        wandb.init(
            project=DEVNULL,
            name="CC-Comp-pL2",
            config=conf_gene_pl2,
            tags=[expe_time],
        ),
    )

    # NOTE - 3.1. GENE w. L2
    conf_gene_l2 = deepcopy(base_config)
    conf_gene_l2["encoding"]["distance"] = "pL2"
    conf_gene_l2["encoding"]["type"] = "gene"
    conf_gene_l2["group"] = "L2"
    validate_json(conf_gene_l2)

    Experiment(
        conf_gene_l2,
        wandb.init(
            project=DEVNULL,
            name="CC-Comp-L2",
            config=conf_gene_l2,
            tags=[expe_time],
        ),
    )

    # NOTE - 3.1. Direct
    conf_direct = deepcopy(base_config)
    conf_direct["encoding"]["type"] = "direct"
    conf_direct["group"] = "direct"
    validate_json(conf_direct)

    Experiment(
        conf_direct,
        wandb.init(
            project=DEVNULL,
            name="CC-Comp-direct",
            config=conf_direct,
            tags=[expe_time],
        ),
    )
