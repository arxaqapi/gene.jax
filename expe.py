from jax import default_backend

from gene.learning import learn_brax_task
from gene.core.distances import Distance_functions
from gene.utils import load_config


if __name__ == "__main__":
    assert default_backend() == "gpu"
    config = load_config("config/brax.json")

    learn_brax_task(
        config=config,
        df=Distance_functions[config["encoding"]["distance"]](),
    )
