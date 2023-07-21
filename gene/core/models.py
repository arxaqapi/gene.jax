from typing import Sequence

import flax.linen as nn


class ReluLinearModel(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x


class ReluLinearModelConf(nn.Module):
    config: dict

    def setup(self):
        self.layers = [
            nn.Dense(feature, name=f"Dense_{i}")
            for i, feature in enumerate(self.config["net"]["layer_dimensions"][1:])
        ]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.layers[-1](x)
        return x


class ReluTanhLinearModel(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        x = nn.tanh(x)
        return x


class ReluTanhLinearModelConf(nn.Module):
    config: dict

    def setup(self):
        self.layers = [
            nn.Dense(feature, name=f"Dense_{i}")
            for i, feature in enumerate(self.config["net"]["layer_dimensions"][1:])
        ]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.layers[-1](x)
        x = nn.tanh(x)
        return x


class TanhLinearModel(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        x = nn.tanh(x)
        return x


class TanhLinearModelConf(nn.Module):
    config: dict

    def setup(self):
        self.layers = [
            nn.Dense(feature, name=f"Dense_{i}")
            for i, feature in enumerate(self.config["net"]["layer_dimensions"][1:])
        ]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.tanh(x)
        return x


Models = {
    "relu_linear": ReluLinearModelConf,
    "relu_tanh_linear": ReluTanhLinearModelConf,
    "tanh_linear": TanhLinearModelConf,
}


def get_model(config: dict) -> nn.Module:
    """Returns the instatianted model fitting the config file.

    Args:
        config (dict): config file of the current run

    Returns:
        nn.Module: Neural network architecture used for evaluation.
    """
    arch = config["net"].get("architecture")
    return Models[arch if arch is not None else "relu_tanh_linear"](config)
