import multiprocessing

class DNGO:
    """ Deep Network for Global Optimization (DNGO)
    """
    def __init__(self):
        pass

    def run(self):
        pass


class SimpleNetwork:
    def __init__(self,
                 data_size: int,
                 feature_dim: int,
                 init_lr: float=0.001,
                 num_epochs: int=3000,
                 hidden1_dim: int=50,
                 hidden2_dim: int=50,
                 hidden3_dim: int=50,
                 output_dim: int=1,
                 activation: str='tanh',
                 train_mode: bool=True):
        import tensorflow as tf
        self.first_layer = tf.keras.models.Sequential([
            K.Dense(hidden1_dim, activation),
            K.Dense(hidden2_dim, activation),
            K.Dense(hidden3_dim, activation)])
        self.last_layer = K.Dense(self.output_dim)

    def build(self, x):
        self.bases = self.first_layer(x)
        return self.last_layer(self.bases)

    def forward(self):
        pass
