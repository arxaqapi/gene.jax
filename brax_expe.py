import wandb

from gene.experiment import Experiment
from gene.utils import fail_if_not_device, load_config, validate_json


if __name__ == "__main__":
    fail_if_not_device()

    config = load_config("config/brax.json")
    validate_json(config)

    exp = Experiment(config)
    stats = exp.run_n(
        seeds=[0, 1, 2, 3, 4, 5],
    )

    exp_wdb_run = wandb.init(project="Brax expe bench test", config=config)

    exp_wdb_run.log(stats)
