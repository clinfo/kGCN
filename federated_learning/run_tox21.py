#!/usr/bin/env python
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

def build_model(max_n_atoms, max_n_types):
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
@click.option('--rounds', default=20, help='the number of updates of the centeral model')
@click.option('--clients', default=2, help='the number of clients')
@click.option('--subsets', default=7, help='the number of subsets')
@click.option('--epochs', default=10, help='the number of training epochs in client traning.')
@click.option('--batchsize', default=32, help='the number of batch size.')
@click.option('--lr', default=0.2, help='learning rate for the central model.')
@click.option('--clientlr', default=0.001, help='learning rate for client models.')
def main(rounds, clients, subsets, epochs, batchsize, lr, clientlr):
    logger = get_logger()
    logger.debug(f'rounds = {rounds}')
    logger.debug(f'clients = {clients}')
    logger.debug(f'subsets = {subsets}')
    logger.debug(f'epochs = {epochs}')
    logger.debug(f'batchsize = {batchsize}')
    logger.debug(f'lr = {lr}')
    logger.debug(f'clientlr = {clientlr}')
    MAX_N_ATOMS = 150
    MAX_N_TYPES = 120
    tox21_train = load_data('train', MAX_N_ATOMS, MAX_N_TYPES, subsets)
    tox21_test = load_data('val', MAX_N_ATOMS, MAX_N_TYPES, subsets)

    # # Pick a subset of client devices to participate in training.
    all_data = [client_data(tox21_train, n, batchsize, epochs) for n in range(subsets)]
    test_data = [client_data(tox21_test, 0, batchsize, epochs),]
    
    def model_fn():
        model = build_model(MAX_N_ATOMS, MAX_N_TYPES)
        return tff.learning.from_keras_model(
            model,
            loss=MultitaskBinaryCrossentropyWithMask(),
            input_spec=all_data[0].element_spec,
            metrics=[
                AUCMultitask(name="auc_task" + str(i), task_number=i) for i in range(12)
            ],
        )
    
    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1),
    )
    state = trainer.initialize()
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    for k in range(clients):
        val_data = [all_data[k],]
        train_data = [d for idx, d in enumerate(all_data) if idx != k]
        logger.debug(f'{k} round ->')
        for round_num in range(rounds):
            state, metrics = trainer.next(state, train_data)
            print(metrics)
            # train_loss = metrics['train']["loss"]
            # train_auc = metrics['train']["auc_task"]
            # logger.debug(f'{round_num:03d} train ===> loss:{train_loss:7.5f}, '
            #              #f'acc:{train_acc:7.5f}, 
            #              f'{train_auc:7.5f},')
        
if __name__ == "__main__":
    main()
