from itertools import product

from dbonas import TrialGenerator

import pytest


def test_trialgenerator_constructor():
    t = TrialGenerator()

def test_trialgenerator_register():
    t = TrialGenerator()
    t.register('batch_size', [10, 12])


@pytest.mark.xfail()
def test_tiralgenerator_with_empty_trials():
    """
    GIVEN: an empty list
    WHEN: try to register it on TiralGenerator
    THEN: throw assertion error
    """
    t = TrialGenerator()
    t.register("batch_size", [])

def test_trialgenerator_register_check_len():
    """
    GIVEN:
    WHEN:
    THEN:
    """
    t = TrialGenerator()
    t.register("batch_size", [1, 2, 3, 4])
    assert len(t) == 4
    # same key is registered
    t.register("batch_size", [1, 2, 3, 4])
    assert len(t) == 4
    t.register("lr", [0.1, 0.2, 0.3, 0.4])
    assert len(t) == 16


def test_trial_index_accesses():
    t = TrialGenerator()
    lrs = [0.1, 0.2, 0.3, 0.4]
    t.register("lr", lrs)
    for i, lr in enumerate(lrs):
        assert t[i].lr == lr
    bs = [128, 256, 512, 1024]
    t.register("batch_size", bs)
    p = product(lrs, bs)
    for i, ele in enumerate(p):
        assert t[i].batch_size == ele[1]
        assert t[i].lr == ele[0]
