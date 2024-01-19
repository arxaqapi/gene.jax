import json
from copy import deepcopy
from typing import Union
from pathlib import Path

from jax import default_backend
import jax.numpy as jnp


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
        if isinstance(base_template[required_key], dict):
            for required_key_2 in base_template[required_key].keys():
                if required_key_2 not in config[required_key].keys():
                    raise ConfigFileIncomplete(
                        f"{required_key_2} (nested level) is missing \
                        from the configuration file."
                    )


def validate_meta_json(config: dict) -> None:
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


def _get_env_sizes(env_name: str):
    from brax.v1.envs import (
        acrobot,
        ant,
        fast,
        fetch,
        grasp,
        half_cheetah,
        hopper,
        humanoid,
        humanoid_standup,
        inverted_double_pendulum,
        inverted_pendulum,
        pusher,
        reacher,
        reacherangle,
        swimmer,
        ur5e,
        walker2d,
    )

    _envs_v1 = {
        "acrobot": acrobot.Acrobot,
        "ant": ant.Ant,
        "fast": fast.Fast,
        "fetch": fetch.Fetch,
        "grasp": grasp.Grasp,
        "halfcheetah": half_cheetah.Halfcheetah,
        "hopper": hopper.Hopper,
        "humanoid": humanoid.Humanoid,
        "humanoidstandup": humanoid_standup.HumanoidStandup,
        "inverted_pendulum": inverted_pendulum.InvertedPendulum,
        "inverted_double_pendulum": inverted_double_pendulum.InvertedDoublePendulum,
        "pusher": pusher.Pusher,
        "reacher": reacher.Reacher,
        "reacherangle": reacherangle.ReacherAngle,
        "swimmer": swimmer.Swimmer,
        "ur5e": ur5e.Ur5e,
        "walker2d": walker2d.Walker2d,
    }
    if env_name not in _envs_v1.keys():
        raise ValueError("env_name not valid")
    env = _envs_v1[env_name]()

    return {"observation_space": env.observation_size, "action_space": env.action_size}


def fix_config_file(config: dict, env_name: Union[str, None] = None) -> dict:
    """Takes a config file and only changes the policy net input and output dimensions

    Args:
        config (dict): base config file to make a modified copy of.
        env_name (Union[str, None], optional): The target base name. Defaults to None.

    Returns:
        dict: the newly created and fixed config file
    """
    env_name = env_name if env_name is not None else config["task"]["environnment"]
    new_config = deepcopy(config)

    new_config["net"]["layer_dimensions"][0] = _get_env_sizes(env_name)[
        "observation_space"
    ]
    new_config["net"]["layer_dimensions"][-1] = _get_env_sizes(env_name)["action_space"]

    return new_config


def meta_save_genome(
    save_path: Union[str, Path],
    wandb_run,
    save_policy: str = "now",
    to_disk: bool = False,
    genome=None,
):
    if to_disk:
        assert genome is not None
        with open(save_path, "wb") as f:
            jnp.save(f, genome)

    wandb_run.save(
        str(save_path),
        base_path=f"{wandb_run.dir}/",
        policy=save_policy,
    )


def make_wdb_subfolder(wandb_run, folder_name: str) -> Path:
    save_path = Path(wandb_run.dir) / folder_name
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path
