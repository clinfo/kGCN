#!/usr/bin/env python
import ast
import datetime
import functools
import logging
import click
import os
import collections

import kgcn.layers as layers
import tensorflow_federated as tff
import tensorflow as tf
from libs.datasets.toxicity import load_data


def pad_adjmat(adj):
    """
    add super node which is connected to all the other nodes
    """
    adj_pad_right = tf.pad(adj, [[0, 0], [0, 0], [0, 0], [0, 1]])
    adj_pad = tf.pad(adj_pad_right, [[0, 0], [0, 0], [0, 1], [0, 0]], constant_values=1)
    return adj_pad


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
    h = layers.GINFL(32, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GINFL(16, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def build_model_gcn_super_node(max_n_atoms, max_n_types):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    # add dummy super node
    padded_adjs = pad_adjmat(input_adjs)
    h = tf.pad(input_features, [[0, 0], [0, 1], [0, 0]]) 
    h = layers.GraphConvFL(32, 1)(h, padded_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(16, 1)(h, padded_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid', name="dense")(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def build_model_gin_super_node(max_n_atoms, max_n_types):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    # add dummy super node
    padded_adjs = pad_adjmat(input_adjs)
    h = tf.pad(input_features, [[0, 0], [0, 1], [0, 0]]) 
    h = layers.GINFL(32, 1)(h, padded_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GINFL(16, 1)(h, padded_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid', name="dense")(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def build_model_gcn(max_n_atoms, max_n_types):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    # for graph
    h = layers.GraphConvFL(32, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(16, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid', name="dense")(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def build_model_gat(max_n_atoms, max_n_types):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    h = layers.GATFL(32)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GATFL(16)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid', name="dense")(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def build_model_diff_pool(max_n_atoms, max_n_types):
    input_adjs = tf.keras.Input(shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(shape=(max_n_atoms, max_n_types), name='features')
    h = layers.GINFL(32, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GINFL(16, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h, adj = layers.DiffPool(16, 5, 1, inner_gnn="gin")(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)


def build_model(model, max_n_atoms, max_n_types):
    if model == 'gcn':
        return build_model_gcn(max_n_atoms, max_n_types)
    elif model == 'gcn_super_node':
        return build_model_gcn_super_node(max_n_atoms, max_n_types)
    elif model == 'gin':
        return build_model_gin(max_n_atoms, max_n_types)
    elif model == 'gin_super_node':
        return build_model_gin_super_node(max_n_atoms, max_n_types)
    elif model == 'gat':
        return build_model_gat(max_n_atoms, max_n_types)
    elif model == 'diff_pool':
        return build_model_diff_pool(max_n_atoms, max_n_types)


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
@click.option('--model', default='gcn', type=click.Choice(['gcn', 'gcn_super_node', 'gin', 'gin_super_node', 'gat', 'diff_pool']), help='support gcn, gin, gat.')
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


def calc_ratios(ratio, subsets):
    if not ratio is None:
        ratio = float(ratio)
        remains_ratio = [(1 - ratio) / (subsets - 1)
                         for _ in range(subsets - 1)]
        ratios = [ratio, ] + remains_ratio
    else:
        ratios = None
    return ratios


def format_metrics(prefix, metrics, round_num):
    acc = metrics["binary_accuracy"]
    loss = metrics["loss"]
    auc = metrics["auc"]
    return f'{prefix}, round, loss, acc, auc ===> {round_num:03d}, {loss:7.5f}, {acc:7.5f}, {auc:7.5f}'


def write_metrics_to_tensorboard(writer, metrics, metric_names, step):
    with writer.as_default():
        for name in metric_names:
            tf.summary.scalar(name, metrics[name], step=step)


def model_summary_as_str(model):
    lines = []
    model.summary(print_fn=lines.append)
    return '    ' + '\n    '.join(lines)


# recode hyperparameters. there may be a better way
def record_experimental_settings(logdir, params, model):
    with tf.summary.create_file_writer(logdir).as_default():
        hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in params.items()]
        tf.summary.text('hyperparameters', tf.stack(hyperparameters), step=0)
        tf.summary.text('model', model_summary_as_str(model), step=0)


def get_log_dir(dataset_name, train_type):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join('logs', 'toxicity', dataset_name, train_type, current_time)


# Wrap a Keras model for use with TFF.
def _model_fn(model, max_n_atoms, max_n_types, input_spec):
    return tff.learning.from_keras_model(
        build_model(model, max_n_atoms, max_n_types),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        input_spec=input_spec)


def federated_learning(rounds, clients, epochs, batchsize, lr, clientlr, model, ratio, dataset_name):
    logger = get_logger()
    subsets = clients + 2
    ratios = calc_ratios(ratio, subsets)
    logger.debug(f'ratios = {ratios}')

    toxicity_train = load_data(
        FL_FLAG=True, dataset_name=dataset_name, n_groups=subsets, subset_ratios=ratios)
    MAX_N_ATOMS = toxicity_train.adj_shape[0]
    MAX_N_TYPES = toxicity_train.feature_shape[1]

    # Pick a subset of client devices to participate in training.
    all_data = [client_data(toxicity_train, n) for n in range(subsets)]
    input_spec = all_data[0].batch(batchsize).element_spec

    logdir = get_log_dir(dataset_name, 'federated')
    record_experimental_settings(
        logdir,
        {'epochs': epochs, 'batchsize': batchsize, "lr": lr, "clientlr": clientlr},
        build_model(model, MAX_N_ATOMS, MAX_N_TYPES))

    model_fn = functools.partial(
        _model_fn, model=model, max_n_atoms=MAX_N_ATOMS, max_n_types=MAX_N_TYPES, input_spec=input_spec)

    # Simulate a few rounds of training with the selected client devices.
    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(clientlr),
    )
    evaluation = tff.learning.build_federated_evaluation(model_fn)

    all_test_metrics = collections.defaultdict(list)
    metric_names = ['binary_accuracy', 'loss', 'auc']

    for k in range(subsets):
        state = trainer.initialize()
        test_data_idx = k
        val_data_idx = (k+1) % subsets
        train_data_indices = [idx for idx in range(subsets) if not idx in [
            test_data_idx, val_data_idx]]
        test_data = [all_data[test_data_idx].batch(batchsize), ]
        val_data = [all_data[val_data_idx].batch(batchsize), ]
        train_data = [repeat_dataset(all_data[idx], batchsize, epochs)
                      for idx in train_data_indices]

        train_writer = tf.summary.create_file_writer(os.path.join(logdir, f'train_{k}'))
        val_writer = tf.summary.create_file_writer(os.path.join(logdir, f'val_{k}'))
        test_writer = tf.summary.create_file_writer(os.path.join(logdir, f'test_{k}'))
        logger.debug(f'{k} round ->')

        for round_num in range(rounds):
            state, metrics = trainer.next(state, train_data)

            logger.debug(format_metrics('train', metrics['train'], round_num))
            write_metrics_to_tensorboard(
                train_writer, metrics['train'], metric_names, round_num)

            val_metrics = evaluation(state.model, val_data)
            logger.debug(format_metrics('val  ', val_metrics, round_num))
            write_metrics_to_tensorboard(val_writer, val_metrics, metric_names, round_num)

        test_metrics = evaluation(state.model, test_data)
        logger.debug(format_metrics('test ', test_metrics, round_num))
        write_metrics_to_tensorboard(test_writer, test_metrics, metric_names, round_num)
        for metric_name in metric_names:
            all_test_metrics[metric_name].append(test_metrics[metric_name])

    logger.debug(f"{clients} >>>> all_test_acc = {all_test_metrics['binary_accuracy']}")
    logger.debug(f"{clients} >>>> all_test_loss = {all_test_metrics['loss']}")
    logger.debug(f"{clients} >>>> all_test_auc = {all_test_metrics['auc']}")


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


def kfold_generator(dataset, k):
    """
    train : test = (k-1) : 1
    """
    dataset_length = tf.data.experimental.cardinality(dataset).numpy()
    shuffled_dataset = dataset.shuffle(
        dataset_length, reshuffle_each_iteration=False)

    def recover(x, y): return y

    for test_idx in range(k):
        def is_test(x, y):
            return x % k == test_idx

        def is_train(x, y):
            return not is_test(x, y)

        train = shuffled_dataset.enumerate().filter(is_train).map(recover)
        test = shuffled_dataset.enumerate().filter(is_test).map(recover)
        yield train, test


def normal_learning(rounds, epochs, batchsize, lr, model_type, dataset_name):
    dataset = load_data(False, dataset_name)
    dataset_length = tf.data.experimental.cardinality(dataset).numpy()
    logdir = get_log_dir(dataset_name, 'normal')

    for idx, (train, test) in enumerate(kfold_generator(dataset, 4)):
        for element in train.take(1):
            MAX_N_TYPES = element[0]['features'].shape[1]
            MAX_N_ATOMS = element[0]['adjs'].shape[1]

        model = build_model(model_type, MAX_N_ATOMS, MAX_N_TYPES)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name="auc")])

        record_experimental_settings(os.path.join(logdir, f"log_{idx}"),
                                     {'epochs': epochs, 'batchsize': batchsize, "lr": lr}, model)

        output_log = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(logdir, f"log_{idx}"), histogram_freq=0, write_graph=True)

        history = model.fit(train.shuffle(dataset_length).batch(batchsize), epochs=epochs,
                            validation_data=test.batch(batchsize), callbacks=[output_log])


if __name__ == "__main__":
    main()
