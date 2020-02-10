from dbonas import Searcher

def test_constructor():
    s = Searcher()


def test_register_trial():
    s = Searcher()
    s.register_trial('batch_size', [128, 256])


def test_search():
    s = Searcher()
    s.register_trial('batch_size', [128, 256])
    def objective(trial):
        print(trial.batch_size)
    pass
