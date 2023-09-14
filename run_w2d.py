from itertools import product as it_product

import wandb

from gene.utils import fail_if_not_device, validate_json, load_config
from gene.core.distances import get_df
from gene.experiment import Experiment

if __name__ == "__main__":
    fail_if_not_device()

    policy_layer_dimensions = (
        [16, 16],
        [32, 32],
        [64, 64],
        [128, 128],
        [32, 32, 32, 32],
    )
    policy_architecture = ("relu_tanh_linear", "tanh_linear")

    experiment_settings = list(
        it_product(
            policy_architecture,
            policy_layer_dimensions,
        )
    )

    for arch, l_dimensions in experiment_settings:
        config = load_config("config/brax.json")
        assert config["task"]["environnment"] == "walker2d"

        config["net"]["architecture"] = arch
        config["net"]["layer_dimensions"] = [17] + l_dimensions + [6]

        validate_json(config)

        with wandb.init(
            project="devnull w2d", name="direct-w2d", config=config
        ) as wandb_run:
            df = get_df(config)()
            Experiment(config, wandb_run, df).run()
