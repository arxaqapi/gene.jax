from typing import Sequence

import flax.linen as nn


class LinearModel(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x


class LinearModelConf(nn.Module):
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


class BoundedLinearModel(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        x = nn.tanh(x)
        return x


class BoundedLinearModelConf(nn.Module):
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
    "linear": LinearModelConf,
    "bounded_linear": BoundedLinearModelConf,
    "tanh_linear": TanhLinearModelConf,
}
