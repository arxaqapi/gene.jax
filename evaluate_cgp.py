"""Takes a certain amount of cgp genomes and evaluates them on different tasks"""
import argparse

import wandb
import jax.numpy as jnp

from gene.experiment import comparison_experiment_cgp
from gene.utils import fail_if_not_device, fix_config_file

from cgpax.analysis.genome_analysis import __save_graph__, __write_readable_program__


def base_to_task(base_config: dict, target_task: str, n_generations: int = 1000):
    """Takes the base of a curriculum and adapts it to the target tasks."""
    copy_base_config = fix_config_file(base_config, target_task)
    copy_base_config["task"]["environnment"] = target_task
    copy_base_config["evo"]["n_generations"] = n_generations
    return copy_base_config


def get_k_best_epoch_metrics(
    run_id: str, reference_metric: str = "training.hc_1000.max_fit", k: int = 5
):
    """Given a `reference_metric` and a `run_id`, get the `k` metrics
    of the epoch maximizing the `reference_metric`.

    Args:
        run_id (str): _description_
        reference_metric (str): _description_
        k (int): top k elements to take
    """
    run = wandb.Api().run(run_id)
    fnet_prop_keys = [
        "training.net_prop.f_expressivity",
        "training.net_prop.f_input_restoration",
        "training.net_prop.f_weight_distribution",
    ]
    metric_history = run.scan_history(keys=[reference_metric, *fnet_prop_keys])

    indexed_metric_history = [{"i": i, **e} for i, e in enumerate(metric_history)]
    filtered_metrics = filter(
        lambda e: e[reference_metric] is not None, indexed_metric_history
    )
    sorted_metrics = sorted(filtered_metrics, key=lambda e: e[reference_metric])
    aggregated_metrics = [
        {**entry_d, "f_net_prop_total": sum([entry_d[key] for key in fnet_prop_keys])}
        for entry_d in reversed(sorted_metrics)
    ]

    return aggregated_metrics[:k]


def get_k_best_genome_ids(
    run_id: str, reference_metric: str = "training.hc_1000.max_fit", k: int = 5
):
    """Given a `reference_metric` and a `run_id`, get the `k` metrics
    of the epoch maximizing the `reference_metric`.

    Args:
        run_id (str): _description_
        reference_metric (str): _description_
        k (int): top k elements to take
    """
    metrics = get_k_best_epoch_metrics(run_id, reference_metric, k)
    return [d["i"] for d in metrics]


def get_file(filepath: str, run):
    with open(run.file(filepath).download(replace=True).name, "rb") as f:
        return jnp.load(f)


def get_genomes_from_run(run_id: str, epoch_ids: list[int]):
    genomes = {}
    run = wandb.Api().run(run_id)
    for epoch in epoch_ids:
        f = f"df_genomes/mg_{epoch}_best_genome.npy"
        genomes[epoch] = get_file(f, run)

    config = run.config
    return genomes, config


def plot_pareto_front(
    metrics: list[dict],
    m1: str = "training.hc_1000.max_fit",
    m2: str = "f_net_prop_total",
):
    import matplotlib.pyplot as plt

    data_m1 = [entry[m1] for entry in metrics]
    data_m2 = [entry[m2] for entry in metrics]

    plt.scatter(data_m1, data_m2)
    plt.title("Pareto front of the k best individuals")
    plt.savefig("pareto_front", dpi=300)


def genome_to_readable(genome, meta_config: dict, filename: str = "test.png"):
    __save_graph__(
        genome=genome,
        config=meta_config["cgp_config"],
        file=f"df_genomes/{filename}",
        input_color="green",
        output_color="red",
    )
    __write_readable_program__(
        genome=genome,
        config=meta_config["cgp_config"],
    )


if __name__ == "__main__":
    fail_if_not_device()

    parser = argparse.ArgumentParser(
        prog="",
        description="Runs the evaluation of the meta evolution best resulting genomes.",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="evaluate-cgp-gene",
        help="Name of the weights and biases project",
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default="sureli",
        help="User used to log on weights and biases.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of best individuals to take.",
    )
    args = parser.parse_args()

    RUN_ID = "sureli/cgp-gene/c4v8hc53"
    tasks = ["halfcheetah", "walker2d", "hopper", "swimmer"]
    extra_tags = [f"k-{args.k}-best"]

    # NOTE - Load cgp genomes to evaluate
    reference_epoch_ids = get_k_best_genome_ids(RUN_ID, k=args.k)
    cgp_genomes_dict, meta_config = get_genomes_from_run(RUN_ID, reference_epoch_ids)

    # NOTE - For each cgp genome, evaluate and compare

    for epoch_id, cgp_genome in cgp_genomes_dict.items():
        # NOTE - evaluate cgp genome on each defined task
        for task in tasks:
            # NOTE - fix config file. Base new config file on w2d curriculum config
            curriculum_config = base_to_task(
                meta_config["curriculum"]["w2d_1000"], task
            )
            # NOTE - add epoch_id to config file
            curriculum_config["epoch_id"] = epoch_id

            comparison_experiment_cgp(
                config=curriculum_config,
                cgp_config=meta_config["cgp_config"],
                # NOTE - Its a mess, we need to encapsulate the genomes correctly
                # cgp_df_genome_archive: dict[int, dict["top_3", list[genomes]]]
                cgp_df_genome_archive={"0": {"top_3": [cgp_genome]}},
                project=args.project,
                entity=args.entity,
                extra_tags=extra_tags,
                seeds=[285033, 99527, 7],
            )
