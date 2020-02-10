from .trial import TrialGenerator

class Searcher:
    def __init__(self):
        self._trials = []
        self.trial_generator = TrialGenerator()
        self.search_algorithm = 'DNGO'

    def register_trial(self, name: str, trial):
        self.trial_generator.register(name, trial)

    def search(self, objective, n_trials: int):
        pass
