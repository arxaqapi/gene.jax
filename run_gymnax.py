import wandb
import jax.random as jrd

from gene.learning import learn_gymnax_task
from gene.utils import fail_if_not_device, load_config, validate_json
from gene.core.distances import get_df


if __name__ == "__main__":
    fail_if_not_device()

    config = load_config("config/acrobot.json")
    validate_json(config)

    df = get_df(config)()

    wdb_run = wandb.init(project="Acrobot", config=config)

    learn_gymnax_task(df, jrd.PRNGKey(0), config, wandb)
