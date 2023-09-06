import wandb

# from gene.meta import meta_learn_cgp
# from gene.meta import meta_learn_cgp_extended
from gene.meta import meta_learn_cgp_corrected
from gene.utils import load_config, fail_if_not_device

from cgpax.jax_functions import available_functions


if __name__ == "__main__":
    fail_if_not_device()

    meta_config = load_config("config/cgp_meta_df_hc.json")
    # meta_config = load_config("config/cgp_meta_df_extended.json")

    meta_config["cgp_config"] = {
        "seed": 6080,
        "problem": "cgp_meta_df",
        "solver": "cgp",
        "mutation": "standard",
        "n_generations": meta_config["evo"]["n_generations"],
        "n_individuals": meta_config["evo"]["population_size"],
        "p_mut_inputs": 0.1,
        "p_mut_functions": 0.1,
        "p_mut_outputs": 0.3,
        "n_nodes": 32,
        "n_functions": len(available_functions),
        "nan_replacement": 0.0,
        "survival": "truncation",
        "selection": {
            "type": "tournament",
            # NOTE - Make this a multiple of the pop size (to check)
            "elite_size": 4,
            "tour_size": 2
        },
    }

    wandb_run = wandb.init(
        project="Meta df benchmarks",
        config=meta_config,
        tags=["cgp", "meta_df_hc", "corrected_f"],
        name="meta-cgp-corrected-df",
    )
    # meta_learn_cgp(meta_config, wandb_run)
    # meta_learn_cgp_extended(meta_config, wandb_run)
    meta_learn_cgp_corrected(meta_config, wandb_run)
