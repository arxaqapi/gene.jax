from gene.learning import learn_brax_task
from gene.core.distances import Distance_functions
from gene.utils import load_config, fail_if_not_device


if __name__ == "__main__":
    fail_if_not_device()

    config = load_config("config/brax.json")

    learn_brax_task(
        config=config,
        df=Distance_functions[config["encoding"]["distance"]](),
    )
