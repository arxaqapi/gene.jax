from gene.tracker import Tracker


class Experiment:
    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self):
        pass

    def run_n(self, seeds: list[int], parallel: bool = False) -> list[Tracker]:
        """Run n experiments, conditioned by the number of seeds provided,
        and returns all statistics.

        Args:
            seed (list[int]): List of seeds used to run the experiments
        """
        if parallel:
            raise NotImplementedError

        for seed in seeds:
            self.config["seed"] = seed
