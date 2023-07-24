"""
Run all benchmarks for the paper.

Naming: 
- Continuous Control benchmarks
- Meta df benchmarks

1. Brax encoding benchmarks 
   ~200 experiments ~ 15/16 h for 1 seed, so 15*5 = 75~80h ~ 3.3 jours
    1. For each seed
        1. for each env:
            1. for each policy_arch
                1.  for each policy_arch_dims
                    1. Direct
                    2. GENE
2. RL algorithms baselines on brax v1
    1. A2C
    2. DDPG
3. meta evolution w. comparable baselines (should be above)
    1. NN based meta-evol
    2. CGP based meta-evol
"""
from datetime import datetime
from itertools import product as it_product

from gene.experiment import Experiment
from gene.utils import fail_if_not_device, validate_json

CONTINUOUS_CONTROL = "Continuous Control benchmarks"
META_DF = "Meta df benchmarks"


if __name__ == "__main__":
    fail_if_not_device()

    seeds = list(range(5))
    brax_envs = {
        "humanoid": {
            "observation_space": 240,
            "action_space": 8,
        },
        "walker2d": {
            "observation_space": 17,
            "action_space": 6,
        },
        "hopper": {
            "observation_space": 11,
            "action_space": 3,
        },
        "ant": {
            "observation_space": 87,
            "action_space": 8,
        },
        "halfcheetah": {
            "observation_space": 18,
            "action_space": 6,
        },
    }
    policy_architecture = ("relu_tanh_linear", "tanh_linear")
    policy_layer_dimensions = (
        [128, 128],
        [32, 32, 32, 32],
    )
    encoding_types = ("direct", "gene")

    experiment_settings = list(
        it_product(
            seeds,
            brax_envs,
            policy_architecture,
            policy_layer_dimensions,
            encoding_types,
        )
    )
    n_total_experiments = len(experiment_settings)

    n_total_experiments_emp = (
        len(brax_envs.keys())
        * len(seeds)
        * len(policy_layer_dimensions)
        * len(policy_architecture)
        * len(encoding_types)
    )
    assert n_total_experiments == n_total_experiments_emp

    config = {
        "seed": 0,
        "evo": {
            "strategy_name": "Sep_CMA_ES",
            "n_generations": 500,
            "population_size": 256,
            "n_evaluations": 1,
        },
        "net": {"layer_dimensions": []},
        "encoding": {"d": 3, "distance": "pL2", "type": "gene"},
        "task": {"environnment": "", "maximize": True, "episode_length": 1000},
        "distance_network": {"layer_dimensions": None},
    }

    print(f"[Log - CCBench] - Running {n_total_experiments} experiments")
    start_time = datetime.now().strftime("%Y.%m.%d_%H:%M")
    for i, (seed, env, arch, dimensions, encoding_type) in enumerate(
        experiment_settings
    ):
        config["seed"] = seed
        config["task"]["environnment"] = env
        config["net"]["architecture"] = arch
        config["net"]["layer_dimensions"] = (
            [brax_envs[env]["observation_space"]]
            + dimensions
            + [brax_envs[env]["action_space"]]
        )
        config["encoding"]["type"] = encoding_type
        print(
            f"[Log - CCBench][{i}] {seed=} | {env=} | {arch=} | {dimensions=} | {config['net']['layer_dimensions']=} | {encoding_type=}"
        )
        # =============================================================
        # NOTE - Start experiment
        validate_json(config)

        exp = Experiment(
            config,
            project_name=CONTINUOUS_CONTROL,
            tags=[f"{start_time}"],
        )
        # do not save intermediate individuals, only start and end
        exp.run(
            seed,
            name=f"{i}-{config['encoding']['type']}-{config['task']['environnment']}",
            save_step=2000,
        )

    print(f"[Log - CCBench] - Finished running {n_total_experiments} experiments")
