import json
from pathlib import Path
from jax import default_backend


class CorrectDeviceNotLoaded(Exception):
    """Raised if the wanted device is not loaded"""

    pass


class ConfigFileIncomplete(Exception):
    """Raised if the given json file is incomplete"""

    pass


def load_config(path: Path):
    with open(path, "r") as f:
        config = json.load(f)
    return config


def fail_if_not_device(device: str = "gpu"):
    """Raises an error if the device used by JAX is not the right one.

    Args:
        device (str, optional): device (str, optional):
            Device to check for. Defaults to "gpu".

    Raises:
        CorrectDeviceNotLoaded: Error to raise.
    """
    default = default_backend()
    if default != device.lower():
        raise CorrectDeviceNotLoaded(f"Current is {default}")


def validate_json(config: dict):
    base_template = {
        "seed": None,
        "evo": {"strategy_name": None, "n_generations": None, "population_size": None},
        "net": {"layer_dimensions": None, "architecture": None},
        "encoding": {"d": None, "distance": None, "type": None},
        "task": {"environnment": None, "maximize": None, "episode_length": None},
    }

    for required_key in base_template.keys():
        if required_key not in config.keys():
            raise ConfigFileIncomplete(
                f"{required_key} (base level) is missing from the configuration file."
            )
        # Level 2 required keys, for nested dict only
        if type(base_template[required_key]) is dict:
            for required_key_2 in base_template[required_key].keys():
                if required_key_2 not in config[required_key].keys():
                    raise ConfigFileIncomplete(
                        f"{required_key_2} (nested level) is missing \
                        from the configuration file."
                    )
