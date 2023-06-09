import wandb

from gene.learn_distance import learn_distance_f_evo


if __name__ == "__main__":
    import json

    with open("config/distance_learning.json") as f:
        config = json.load(f)

    wdb_run = wandb.init(
        project="Distance-tests",
        config=config,
    )

    fit, center = learn_distance_f_evo(config, wdb_run)

    print(sorted(fit, reverse=True)[:3])
    print(f"{center=}")
