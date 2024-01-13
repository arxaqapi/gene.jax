import wandb
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cgpax.analysis.genome_analysis import __save_graph__, __write_readable_program__


def load_run(run_id: str):
    api = wandb.Api()
    run = api.run(run_id)
    return run


def get_file(filepath: str, run):
    with open(run.file(filepath).download(replace=True).name, "rb") as f:
        genome = jnp.load(f)
    return genome


def get_k_best_epochs(run, metric: str = "training.hc_1000.max_fit", k: int = 5):
    fnet_prop_keys = [
        "training.net_prop.f_expressivity",
        "training.net_prop.f_input_restoration",
        "training.net_prop.f_weight_distribution",
    ]
    metric_history = run.scan_history(keys=[metric, *fnet_prop_keys])

    indexed_metric_history = [{"i": i, **e} for i, e in enumerate(metric_history)]
    filtered_metrics = filter(lambda e: e[metric] is not None, indexed_metric_history)
    sorted_metrics = sorted(filtered_metrics, key=lambda e: e[metric])
    aggregated_metrics = [
        {**entry_d, "f_net_prop_total": sum([entry_d[key] for key in fnet_prop_keys])}
        for entry_d in reversed(sorted_metrics)
    ]

    return aggregated_metrics[:k]


def plot_pareto_front(
    data: list[dict], m1: str = "training.hc_1000.max_fit", m2: str = "f_net_prop_total"
):
    data_m1 = [entry[m1] for entry in data]
    data_m2 = [entry[m2] for entry in data]

    plt.scatter(data_m1, data_m2)
    plt.title("Pareto front of the k best individuals")
    plt.savefig("pareto_front", dpi=300)


if __name__ == "__main__":
    RUN_ID = "sureli/cgp-gene/c4v8hc53"

    run = load_run(RUN_ID)

    k_best = get_k_best_epochs(run, k=30)
    plot_pareto_front(k_best)
