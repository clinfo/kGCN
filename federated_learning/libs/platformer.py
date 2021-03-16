import tensorflow_federated as tff

from .utils import get_logger


class Platformer:
    ''' 
    Platformer provides interfaces of hyperparameter tuning and 
    k-fold cross validation training.
    '''
    def __init__(self, trainer, model_fn, datasets, metric_names = ['loss',], kfold=True, logdir=None, logger=None):
        self.trainer = trainer
        self.evaluation = tff.learning.build_federated_evaluation(model_fn)
        self.model_fn = model_fn
        self.datasets = datasets        
        self.metric_names = metric_names
        self.kfold = kfold
        if logdir is not None:
            self.writer = tf.summary.create_file_writer(logdir)            
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = 'logs/fl/' + current_time
            self.writer = tf.summary.create_file_writer(logdir)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_logger('Federated Learning')
        self.n_subsets = len(self.datasets)

    def run(self):
        evaluation = tff.learning.build_federated_evaluation(model_fn)

        for k in range(self.n_subsets):
            self.logger("{k} round training")
            state = self.trainer.initialize()
            train_data, val_data, test_data = self._create_dataset(k)
            state, metrics = self.trainer.next(state, train_data)
            val_metrics = self.evaluation(state.model, val_data)            
            train_loss = metrics['train']["loss"]
            val_loss = val_metrics["loss"]
            with self.writer.as_default():
                tf.summary.scalar(f'train_loss{k}', train_loss, step=round_num)
                tf.summary.scalar(f'train_auc{k}', train_auc, step=round_num)
                tf.summary.scalar(f'val_loss{k}', val_loss, step=round_num)
                tf.summary.scalar(f'val_auc{k}', val_auc, step=round_num)                


    def _create_dataset(self, test_data_idx):
        val_data_idx = test_data_idx - 1 if test_data_idx == (self.n_subsets - 1) \
            else test_data_idx + 1
        test_data = [all_data[test_data_idx],]
        val_data = [all_data[val_data_idx],]
        train_data = [d for idx, d in enumerate(all_data) if not idx in [test_data_idx, val_data_idx]]
        return train_data, val_data, test_data

    def tuning(self):
        pass
    
