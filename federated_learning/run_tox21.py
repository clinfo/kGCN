#!/usr/bin/env python
import ast
import functools
import logging
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import tensorflow_federated as tff
import kgcn.layers as layers

from datasets.tox21 import load_data


def get_logger(level='DEBUG'):
    FORMAT = '%(asctime)-15s - %(pathname)s - %(funcName)s - L%(lineno)3d ::: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger

def client_data(source, n, batch_size, epochs):
    return source.create_tf_dataset_for_client(source.client_ids[n]).repeat(epochs).batch(batch_size)


def build_model_gcn(max_n_atoms, max_n_types):
    input_features = tf.keras.Input(shape=(max_n_atoms, max_n_types), name="features")
    input_adjs = tf.keras.Input(shape=(1, max_n_atoms, max_n_atoms), name="adjs", sparse=False)
    input_mask_label = tf.keras.Input(shape=(12), name="mask_label")
    h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(12, tf.nn.sigmoid, input_shape=[64])(h)
    return keras.Model(
        inputs=[input_features, input_adjs, input_mask_label],
        outputs=tf.stack([logits, input_mask_label]),
    )


def build_model_gin(max_n_atoms, max_n_types):
    input_features = tf.keras.Input(shape=(max_n_atoms, max_n_types), name="features")
    input_adjs = tf.keras.Input(shape=(1, max_n_atoms, max_n_atoms), name="adjs", sparse=False)
    input_mask_label = tf.keras.Input(shape=(12), name="mask_label")
    h = layers.GINFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GINFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(12, tf.nn.sigmoid, input_shape=[64])(h)
    return keras.Model(
        inputs=[input_features, input_adjs, input_mask_label],
        outputs=tf.stack([logits, input_mask_label]),
    )


class MultitaskBinaryCrossentropyWithMask(keras.losses.Loss):
    def call(self, y_true, model_out):
        logits = model_out[0]
        masks = model_out[1]
        losses = []
        for task in range(12):
            mask = tf.cast(masks[:, task], tf.bool)
            y_true_masked = tf.boolean_mask(y_true[:, task], mask)
            logits_masked = tf.boolean_mask(logits[:, task], mask)
            loss = tf.keras.losses.binary_crossentropy(
                y_true_masked, logits_masked, from_logits=False
            )
            losses.append(loss)
        loss = tf.stack(losses)
        return loss


class AUCMultitask(keras.metrics.AUC):
    def __init__(self, name="auc_multitask", task_number=0, **kwargs):
        super(AUCMultitask, self).__init__(name=name, **kwargs)
        self.task_number = task_number

    def update_state(self, y_true, y_pred, sample_weight=None):
        model_out = y_pred
        logits = model_out[0]
        masks = model_out[1]
        losses = []
        mask = tf.cast(masks[:, self.task_number], tf.bool)
        y_true_masked = tf.boolean_mask(y_true[:, self.task_number], mask)
        logits_masked = tf.boolean_mask(logits[:, self.task_number], mask)
        super(AUCMultitask, self).update_state(y_true_masked, logits_masked)


@click.command()
@click.option('--rounds', default=10, help='the number of updates of the centeral model')
@click.option('--clients', default=2, help='the number of clients')
@click.option('--subsets', default=7, help='the number of subsets')
@click.option('--epochs', default=10, help='the number of training epochs in client traning.')
@click.option('--batchsize', default=32, help='the number of batch size.')
@click.option('--lr', default=0.2, help='learning rate for the central model.')
@click.option('--clientlr', default=0.001, help='learning rate for client models.')
@click.option('--model', default='gcn', type=click.choice(['gcn', 'gin']),
              help='support gcn or gin.')
@click.option('--ratio', default=None, help='set ratio of the biggest dataset in total datasize.' + \
              ' Other datasets are equally divided. (0, 1)')
@click.option('--task',  default=None, 
              type=click.choice(['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                                 'SR-HSE', 'SR-MMP', 'SR-p53']), 
              default=None, help='set ratio of the biggest dataset in total datasize.')
def main(rounds, clients, subsets, epochs, batchsize, lr, clientlr, model, ratio, task):
    logger = get_logger()
    logger.debug(f'rounds = {rounds}')
    logger.debug(f'clients = {clients}')
    logger.debug(f'subsets = {subsets}')
    logger.debug(f'epochs = {epochs}')
    logger.debug(f'batchsize = {batchsize}')
    logger.debug(f'lr = {lr}')
    logger.debug(f'clientlr = {clientlr}')
    logger.debug(f'model = {model}')
    logger.debug(f'ratio = {ratio}')
    logger.debug(f'task = {task}')
    if not model in ['gcn', 'gin']:
        raise Exception(f'not supported model. {model}')
    MAX_N_ATOMS = 150
    MAX_N_TYPES = 120
    if isinstance(ratio, str):
        ratio = ast.literal_eval(ratio)
    if not ratio is None:
        ratio = float(ratio)
        remains_ratio = [(1 - ratio) / (subsets - 1) for _ in range(subsets - 1)]
        ratios = [ratio, ] + remains_ratio
    else:
        ratios = None
    tox21_train = load_data('train', MAX_N_ATOMS, MAX_N_TYPES, subsets, ratios, task)
    #tox21_test = load_data('val', MAX_N_ATOMS, MAX_N_TYPES, subsets)

    # # Pick a subset of client devices to participate in training.
    all_data = [client_data(tox21_train, n, batchsize, epochs) for n in range(subsets)]
    #test_data = [client_data(tox21_test, 0, batchsize, epochs),]

    def _model_fn(model):
        if model == "gcn":
            model = build_model_gcn(MAX_N_ATOMS, MAX_N_TYPES)
        elif model == "gin":
            model = build_model_gin(MAX_N_ATOMS, MAX_N_TYPES)
        return tff.learning.from_keras_model(
            model,
            loss=MultitaskBinaryCrossentropyWithMask(),
            input_spec=all_data[0].element_spec,
            metrics=[
                AUCMultitask(name="auc_task" + str(i), task_number=i) for i in range(12)
            ],
        )
    model_fn = functools.partial(_model_fn, model=model)    
    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1),
    )
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    all_test_loss = []
    all_test_auc = []
    for k in range(subsets):
        state = trainer.initialize()
        test_data_idx = k
        val_data_idx = k - 1 if k == (subsets - 1) else k + 1
        test_data = [all_data[test_data_idx],]
        val_data = [all_data[val_data_idx],]
        train_data = [d for idx, d in enumerate(all_data) if not idx in [test_data_idx, val_data_idx]]
        logger.debug(f'{k} round ->')

        for round_num in range(rounds):
            train_aucs = []
            val_aucs = []
            state, metrics = trainer.next(state, train_data)
            train_loss = metrics['train']["loss"]
            for i in range(12):
                train_aucs.append(metrics['train'][f"auc_task{i}"])
            logger.debug(f' train, round, loss, acus ===> {round_num:03d}, {train_loss:7.5f}, '
                         f', {train_aucs}')
            val_metrics = evaluation(state.model, val_data)
            val_loss = val_metrics["loss"]
            for i in range(12):
                val_aucs.append(val_metrics[f"auc_task{i}"])
            logger.debug(f'val, round, loss, auc ===> {round_num:03d}, {val_loss:7.5f}, '
                         f' {val_aucs},')
        test_metrics = evaluation(state.model, test_data)
        test_loss = test_metrics["loss"]
        test_aucs = []
        for i in range(12):
            test_aucs.append(val_metrics[f"auc_task{i}"])
        logger.debug(f'test, round, loss, auc ===> {k}, {round_num:03d}, {test_loss:7.5f}, '
                     f' {test_aucs},')
        all_test_loss.append(test_loss)
        all_test_auc.append(test_aucs)
    logger.debug(f'{clients} >>>> all_test_loss = {all_test_loss}')
    logger.debug(f'{clients} >>>> all_test_auc = {all_test_auc}')
        
if __name__ == "__main__":
    main()
