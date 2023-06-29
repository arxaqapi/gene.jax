import unittest

from gene.utils import validate_json, load_config, ConfigFileIncomplete


class TestJsonValidation(unittest.TestCase):
    def test_brax_config_file(self):
        config = load_config("config/brax.json")

        self.assertIsNone(validate_json(config))

    def test_raises_missing_key_base_level(self):
        config = {
            "evo": {
                "strategy_name": None,
                "n_generations": None,
                "population_size": None,
            },
            "net": {"layer_dimensions": None, "architecture": None},
            "encoding": {"d": None, "distance": None, "type": None},
            "task": {"environnment": None, "maximize": None, "episode_length": None},
        }
        with self.assertRaises(ConfigFileIncomplete):
            validate_json(config)
