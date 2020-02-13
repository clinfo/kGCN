import random

import pytest

from dbonas import Searcher

def test_constructor():
    s = Searcher()

def test_register_trial():
    s = Searcher()
    s.register_trial('batch_size', [128, 256, 512, 1024])

def test_initialize_trials():
    s = Searcher()
    s.register_trial('batch_size', [128, 256, 512, 1024])
    assert list(range(4)) == s.initialize_trials()

def test_reinitialize_trials():
    s = Searcher()
    s.register_trial('batch_size', [128, 256, 512, 1024])
    assert list(range(4)) == s.initialize_trials()
    s.register_trial('batch_size', [128, 256, 512, 1024, 2048])
    assert list(range(5)) == s.initialize_trials()
    s.register_trial('lr', [0.1, 0.2, 0.3, 0.4, 0.5])
    assert list(range(25)) == s.initialize_trials()


def test_random_search():
    random.seed(0)
    s = Searcher()
    s.register_trial('batch_size', [128, 256, 512])
    s.register_trial('lr', [0.1, 0.2, 0.3, 0.4, 0.5])
    s.initialize_trials()
    def objective(trial):
        return 0
    remain_indices = s.random_search(objective, 10)
    assert len(remain_indices) == 5
    assert remain_indices == [1, 5, 9, 10, 11]

def test_search():
    s = Searcher()
    s.register_trial('batch_size', [128, 256])
    def objective(trial):
        print(trial.batch_size)
    pass

@pytest.mark.xfail
def test_search_fail_case_without_random_search():
    s = Searcher()
    s.register_trial('batch_size', [128, 256])
    s.initialize_trials()
    def objective(trial):
        print(trial.batch_size)
    s.search(objective, 10)

def test_search_normal_search():
    s = Searcher()
    s.register_trial('batch_size', [128, 256, 512, 1024])
    s.register_trial('dim1', [10, 15, 20, 40])
    s.register_trial('lr', [0.1, 0.2, 0.3, 0.4])
    s.initialize_trials()
    def objective(trial):
        print(trial.batch_size)
    s.random_search(objective, 10)
    s.search(objective, 10)
