from gene.experiment import Experiment
from gene.utils import fail_if_not_device, load_config, validate_json


if __name__ == "__main__":
    fail_if_not_device()

    config = load_config("config/brax.json")
    validate_json(config)

    exp = Experiment(config, "REMOVE ME")
    stats = exp.run_n(seeds=[0, 1])
