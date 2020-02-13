import typing
import random

class Trial:
    def __init__(self):
        self._elements = {}

    def __setattr__(self, name: str, value):
        super.__setattr__(self, name, value)
        self._elements[name] = value

    def __str__(self):
        o = "{"
        for k, v in self._elements.items():
            if k == '_elements':
                continue
            o += f"{k}: {v}, "
        o = o[:-2] # remove the last comma.
        o += "}"
        return o


class TrialGenerator:
    def __init__(self):
        self._registered: typing.Dict[str, list] = {}
        self._registered_length: typing.Dict[str, int] = {}
        self.trial = Trial()
        self._len = 1

    def register(self, name: str, trials: typing.List) -> None:
        assert len(trials) != 0, "can't accept empty trials."
        if name in self._registered.keys():
            print(f"{name} is already registered")
            return
        self._registered[name] = trials
        self._registered_length[name] = len(trials)
        setattr(self.trial, name, None)
        self._len *= len(trials)

    def __getitem__(self, i: int) -> Trial:
        indexes: typing.Dict[str, int] = {}
        for idx, (k, n) in enumerate(self._registered_length.items()):
            if (idx + 1) == len(self._registered):
                indexes[k] = i % n
            else:
                indexes[k] = i // n
        for k, n in indexes.items():
            setattr(self.trial, k, self._registered[k][n])
        return self.trial

    def __len__(self):
        if len(self._registered) == 0:
            return 0
        return self._len

    def __str__(self):
        o = ""
        for k, i in self._registered.values():
            o += f"{k} : {i}\n"
        return o
