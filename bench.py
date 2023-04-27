import jax.random as jrd
import jax_smi

from run import run
import json
import copy

if __name__ == "__main__":
    with open("config/base_bench.json", "r") as f:
        config_gene = json.load(f)

    config_direct = copy.deepcopy(config_gene)
    config_direct["encoding"]["type"] = "direct"

    jax_smi.initialise_tracking(interval=0.5, dir_prefix="profiles/direct_trace")
    # run(config_gene, jrd.PRNGKey(0))
    run(config_direct, jrd.PRNGKey(0))
