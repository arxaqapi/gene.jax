import json
from jax import default_backend


class CorrectDeviceNotLoaded(Exception):
    """Raised if the wanted device is not loaded"""

    pass


class ConfigFileIncomplete(Exception):
    """Raised if the given json file is incomplete"""

    pass


def load_config(path: str):
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


def validate_json(config: dict) -> None:
    """Validates the json format of the passed configuration file.
    Only checks that all fields are present, and not their type.

    Args:
        config (dict): The configuration file to validate

    Raises:
        ConfigFileIncomplete: Is the configuration file is incomplete
            this error will be raised in response
    """
    base_template = {
        "seed": None,
        "evo": {
            "strategy_name": None,
            "n_generations": None,
            "population_size": None,
            "n_evaluations": None,
        },
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


def validate_meta_json(conig: dict) -> None:
    """Validates the json format of the passed meta configuration file.
    Only checks that all fields are present, and not their type.

    Args:
        config (dict): The configuration file to validate

    Raises:
        ConfigFileIncomplete: Is the configuration file is incomplete
            this error will be raised in response
    """
    raise NotImplementedError


def min_max_scaler(x):
    "Brings value to the [0, 1] range"
    x_min = x.min()
    return (x - x_min) / ((x.max() - x_min) + 1e-6)
