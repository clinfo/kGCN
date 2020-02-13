import pathlib
import random
import typing

from .trial import (Trial,
                    TrialGenerator)

from .logger import get_logger
from .dngo import DNGO

class Searcher:
    def __init__(self, logger=None):
        self.trial_generator = TrialGenerator(logger)
        self.search_algorithm = 'DNGO'
        self._trials_indices: typing.List[int] = []
        self._searched_trial_indices: typing.List[int] = []
        self.results: typing.Dict[int, float] = {}
        if logger is None:
            self.logger = get_logger("Searcher")
        else:
            self.logger = logger

    def register_trial(self, name: str, trial: list):
        self.trial_generator.register(name, trial)

    def initialize_trials(self) -> typing.List[int]:
        self._trials_indices = list(range(len(self.trial_generator)))
        self._searched_trial_indices: typing.List[int] = []
        return self._trials_indices

    def random_search(self, objective: typing.Callable[[Trial], float], n_trials: int) -> typing.List[int]:
        trial_indices = random.sample(self._trials_indices, n_trials)
        for i in trial_indices:
            self.results[i] = objective(self.trial_generator[i])
            self._trials_indices.remove(i)
        self._searched_trial_indices += trial_indices
        return self._trials_indices # return remained trials

    def search(self, objective: typing.Callable[[Trial], float], n_trials: int):
        assert len(self._searched_trial_indices) != 0, 'Before do this, you have to run random search'

    def __len__(self):
        return len(self.trial_generator)

    @staticmethod
    def read_config(path: pathlib.Path or str) -> Self:
        pass

    @staticmethod
    def _read_toml(path: pathlib.Path) -> Self:
        pass

    @staticmethod
    def _read_yaml(path: pathlib.Path) -> Self:
        pass

    def dump(self, path: pathlib.Path or str):
        pass

    def _dump_yaml(self, path: pathlib.Path):
        pass

    def _dump_toml(self, path: pathlib.Path):
        pass
