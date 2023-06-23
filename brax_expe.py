import wandb

from gene.experiment import Experiment
from gene.utils import fail_if_not_device, load_config


if __name__ == "__main__":
    fail_if_not_device()

    config = load_config("config/brax.json")

    exp = Experiment(config)
    exp.run_n(
        exp_wdb_run=wandb.init(project="Brax expe bench test", config=config),
        seeds=[0, 1, 2, 3, 4, 5],
    )
