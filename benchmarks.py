"""
Run all benchmarks for the paper.

Naming: 
- Continuous Control benchmarks
- Meta df benchmarks

1. Brax encoding benchmarks ~200 experiments
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

    n_total_experiments = (
        len(brax_envs.keys())
        * len(seeds)
        * len(policy_layer_dimensions)
        * len(policy_architecture)
        * len(encoding_types)
    )

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
    i = 0
    for seed in seeds:
        config["seed"] = seed

        for env in brax_envs.keys():
            config["task"]["environnment"] = env

            for arch in policy_architecture:
                config["net"]["architecture"] = arch

                for dimensions in policy_layer_dimensions:
                    o_s = brax_envs[env]["observation_space"]
                    a_s = brax_envs[env]["action_space"]
                    config["net"]["layer_dimensions"] = [o_s] + dimensions + [a_s]

                    for encoding_type in encoding_types:
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
                            name=f"{i}-{config['encoding']['type']}-full",
                            save_step=2000,
                        )

                        # =============================================================
                        i += 1

    print(f"[Log - CCBench] - Finished running {i} experiments")
