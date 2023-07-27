import wandb

from gene.meta import meta_learn_nn
from gene.utils import load_config, fail_if_not_device


if __name__ == "__main__":
    fail_if_not_device()
    config = load_config("config/nn_meta_df.json")

    wandb_run = wandb.init(project="Meta df benchmarks", config=config, tags=["nn"])

    meta_learn_nn(config, wandb_run)