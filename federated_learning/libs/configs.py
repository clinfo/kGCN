

class Config:
    def load(self):
        pass

    def save(self):
        pass

class TrainConfig(Config):
    def __init__(self):
        self.lr = lr
        self.clientlr = lr

class HyperParameterSearchConfig(Config):
    def __init__(self):
        pass
