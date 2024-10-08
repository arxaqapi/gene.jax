import time
import wandb
from copy import deepcopy

from gene.utils import fail_if_not_device, load_config
from gene.experiment import comparison_experiment
from gene.meta import meta_learn_nn_corrected

DEVNULL = "devnull"

if __name__ == "__main__":
    fail_if_not_device()

    expe_time = int(time.time())
    extra_tags = ["full-tanh-test"]

    # NOTE - 1. Meta nn
    meta_nn_config = load_config("config/nn_meta_df_corrected.json")
    # meta_nn_config = load_config("config/nn_meta_df_corrected_w2d.json")
    meta_nn_config["group"] = "meta"
    with wandb.init(
        project=DEVNULL,
        name="CC-Comp-meta",
        config=meta_nn_config,
        tags=[f"{expe_time}"] + extra_tags,
    ) as meta_nn_wdb:
        _, best_df_genome, nn_df_model = meta_learn_nn_corrected(
            meta_nn_config,
            meta_nn_wdb,
            beta=1,
        )

    comparison_experiment(
        config=deepcopy(next(iter(meta_nn_config["curriculum"].values()))),
        nn_df_genome=best_df_genome,
        nn_df_model_config={"net": meta_nn_config["net"]},
        project=DEVNULL,
        expe_time=expe_time,
        extra_tags=extra_tags,
    )
