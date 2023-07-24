from gene.meta import meta_learn_cgp
from gene.utils import load_config, fail_if_not_device

from cgpax.jax_functions import available_functions


if __name__ == "__main__":
    fail_if_not_device()

    meta_config = load_config("config/cgp_meta_df.json")

    cgp_config = {
        "seed": 0,
        "problem": "cgp_meta_df",
        "n_generations": 200,
        "n_individuals": 16,
        "elite_size": 5,
        "p_mut_inputs": 0.1,
        "p_mut_functions": 0.1,
        "p_mut_outputs": 0.3,
        "n_nodes": 10,
        "n_functions": len(available_functions),
        "constrain_outputs": True,
        "nan_replacement": 10e10,
    }

    assert cgp_config["n_generations"] == meta_config["evo"]["n_generations"]
    assert cgp_config["n_individuals"] == meta_config["evo"]["population_size"]

    meta_learn_cgp(meta_config, cgp_config)
