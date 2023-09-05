from gene.experiment import meta_comparison_experiment
from gene.utils import load_config, validate_json, fail_if_not_device


if __name__ == "__main__":
    fail_if_not_device()

    config = load_config("config/brax.json")
    validate_json(config)

    meta_comparison_experiment(config)
