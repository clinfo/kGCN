#!/usr/bin/env python
import ast
import datetime
import functools
import logging
import click
import os

import kgcn.layers as layers
import tensorflow_federated as tff
import tensorflow as tf
from libs.datasets.toxicity import load_data


def get_logger(level='DEBUG'):
    FORMAT = '%(asctime)-15s - %(pathname)s - %(funcName)s - L%(lineno)3d ::: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger


def build_model_gin(max_n_atoms, max_n_types):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    h = layers.GINFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GINFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def build_model_gcn(max_n_atoms, max_n_types):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    # for graph
    h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid', name="dense")(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def client_data(source, n):
    return source.create_tf_dataset_for_client(source.client_ids[n])


def repeat_dataset(client_dataset, batch_size, epochs):
    dataset_length = tf.data.experimental.cardinality(client_dataset).numpy()
    return client_dataset.shuffle(dataset_length, reshuffle_each_iteration=True).repeat(epochs).batch(batch_size)


@click.command()
@click.option('--federated', is_flag=True)
@click.option('--rounds', default=10, help='the number of updates of the central model')
@click.option('--clients', default=2, help='the number of clients')
@click.option('--epochs', default=10, help='the number of training epochs in client traning.')
@click.option('--batchsize', default=32, help='the number of batch size.')
@click.option('--lr', default=0.2, help='learning rate for the central model.')
@click.option('--clientlr', default=0.001, help='learning rate for client models.')
@click.option('--model', default='gcn', type=click.Choice(['gcn', 'gin']), help='support gcn or gin.')
@click.option('--ratio', default=None, help='set ratio of the biggest dataset in total datasize.' +
              ' Other datasets are equally divided. (0, 1)')
@click.option('--dataset_name', type=click.Choice(['Benchmark', 'NTP_PubChem_Bench.20201106', 'Ames_S9_minus.20201106']),
              default='NTP_PubChem_Bench.20201106', help='set dataset name')
def main(federated, rounds, clients, epochs, batchsize, lr, clientlr, model, ratio, dataset_name):
    logger = get_logger()
    subsets = clients + 2
    logger.debug(f'federated = {federated}')
    logger.debug(f'rounds = {rounds}')
    logger.debug(f'clients = {clients}')
    logger.debug(f'subsets = {subsets}')
    logger.debug(f'epochs = {epochs}')
    logger.debug(f'batchsize = {batchsize}')
    logger.debug(f'lr = {lr}')
    logger.debug(f'clientlr = {clientlr}')
    logger.debug(f'model = {model}')
    logger.debug(f'ratio = {ratio}')
    logger.debug(f'dataset_name = {dataset_name}')

    if federated:
        federated_learning(rounds, clients, epochs, batchsize,
                           lr, clientlr, model, ratio, dataset_name)
    else:
        normal_learning(rounds, epochs, batchsize, lr, model, dataset_name)


def federated_learning(rounds, clients, epochs, batchsize, lr, clientlr, model, ratio, dataset_name):
    logger = get_logger()
    subsets = clients + 2
    MAX_N_ATOMS = 250
    MAX_N_TYPES = 100
    if not ratio is None:
        ratio = float(ratio)
        remains_ratio = [(1 - ratio) / (subsets - 1)
                         for _ in range(subsets - 1)]
        ratios = [ratio, ] + remains_ratio
    else:
        ratios = None
    logger.debug(f'ratios = {ratios}')
    toxicity_train = load_data(FL_FLAG=True, dataset_name=dataset_name, max_n_atoms=MAX_N_ATOMS,
                               max_n_types=MAX_N_TYPES, n_groups=subsets, subset_ratios=ratios)
    # # Pick a subset of client devices to participate in training.
    all_data = [client_data(toxicity_train, n) for n in range(subsets)]
    input_spec = all_data[0].batch(batchsize).element_spec

    # Wrap a Keras model for use with TFF.
    def _model_fn(model):
        if model == 'gcn':
            model = build_model_gcn(MAX_N_ATOMS, MAX_N_TYPES)
        elif model == 'gin':
            model = build_model_gin(MAX_N_ATOMS, MAX_N_TYPES)
        return tff.learning.from_keras_model(
            model,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
            input_spec=input_spec)

    model_fn = functools.partial(_model_fn, model=model)

    # Simulate a few rounds of training with the selected client devices.
    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(clientlr),
    )

    evaluation = tff.learning.build_federated_evaluation(model_fn)

    all_test_acc = []
    all_test_loss = []
    all_test_auc = []
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', 'toxicity', dataset_name,
                          'federated', current_time)
    writer = tf.summary.create_file_writer(logdir)

    for k in range(subsets):
        state = trainer.initialize()
        test_data_idx = k
        val_data_idx = k - 1 if k == (subsets - 1) else k + 1
        test_data = [all_data[test_data_idx].batch(batchsize), ]
        val_data = [all_data[val_data_idx].batch(batchsize), ]
        train_data = [repeat_dataset(d, batchsize, epochs) for idx, d in enumerate(all_data) if not idx in [
            test_data_idx, val_data_idx]]
        logger.debug(f'{k} round ->')

        for round_num in range(rounds):
            state, metrics = trainer.next(state, train_data)
            train_acc = metrics['train']["binary_accuracy"]
            train_loss = metrics['train']["loss"]
            train_auc = metrics['train']["auc"]
            logger.debug(
                f'train, round, loss, acc, auc  ===> {round_num:03d}, {train_loss:7.5f}, {train_acc:7.5f}, {train_auc:7.5f},')
            val_metrics = evaluation(state.model, val_data)
            val_acc = val_metrics["binary_accuracy"]
            val_loss = val_metrics["loss"]
            val_auc = val_metrics["auc"]
            logger.debug(
                f'val, round, loss, acc, auc ===> {round_num:03d}, {val_loss:7.5f}, {val_acc:7.5f}, {val_auc:7.5f},')
            with writer.as_default():
                tf.summary.scalar(f'train_loss{k}', train_loss, step=round_num)
                tf.summary.scalar(f'train_auc{k}', train_auc, step=round_num)
                tf.summary.scalar(f'val_loss{k}', val_loss, step=round_num)
                tf.summary.scalar(f'val_auc{k}', val_auc, step=round_num)

        test_metrics = evaluation(state.model, test_data)
        test_acc = test_metrics["binary_accuracy"]
        test_loss = test_metrics["loss"]
        test_auc = test_metrics["auc"]
        logger.debug(f'test, round, loss, acc, auc  ===> {round_num:03d}, {test_loss:7.5f}, '
                     f'{test_acc:7.5f}, {test_auc:7.5f},')

        with writer.as_default():
            tf.summary.scalar(f'test_loss{k}', test_loss, step=round_num)
            tf.summary.scalar(f'test_auc{k}', test_auc, step=round_num)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        all_test_auc.append(test_auc)

    logger.debug(f'{clients} >>>> all_test_acc = {all_test_acc}')
    logger.debug(f'{clients} >>>> all_test_loss = {all_test_loss}')
    logger.debug(f'{clients} >>>> all_test_auc = {all_test_auc}')


def split_train_test_val(dataset):
    """
    splits the dataset into train, test, val\n
    train : test : val = 8 : 1 : 1
    """
    dataset_length = tf.data.experimental.cardinality(dataset).numpy()
    shuffled_dataset = dataset.shuffle(
        dataset_length, reshuffle_each_iteration=False)

    def recover(x, y): return y

    def is_test(x, y):
        return x % 10 == 0

    def is_val(x, y):
        return x % 10 == 1

    def is_train(x, y):
        return not (is_test(x, y) or is_val(x, y))

    train = shuffled_dataset.enumerate().filter(is_train).map(recover)
    test = shuffled_dataset.enumerate().filter(is_test).map(recover)
    val = shuffled_dataset.enumerate().filter(is_val).map(recover)
    return train, test, val


def normal_learning(rounds, epochs, batchsize, lr, model, dataset_name):
    MAX_N_ATOMS = 250
    MAX_N_TYPES = 100

    dataset = load_data(False, dataset_name, MAX_N_ATOMS, MAX_N_TYPES)
    dataset_length = tf.data.experimental.cardinality(dataset).numpy()
    train, test, val = split_train_test_val(dataset)

    if model == 'gcn':
        model = build_model_gcn(MAX_N_ATOMS, MAX_N_TYPES)
    elif model == 'gin':
        model = build_model_gin(MAX_N_ATOMS, MAX_N_TYPES)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

    model.summary()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', 'toxicity',
                          dataset_name, 'normal', current_time)
    writer = tf.summary.create_file_writer(logdir)

    output_log = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=0, write_graph=True)

    model.fit(train.shuffle(dataset_length).batch(batchsize), epochs=epochs,
              validation_data=test.batch(batchsize), callbacks=[output_log])

    model.evaluate(test.batch(1))


if __name__ == "__main__":
    main()
