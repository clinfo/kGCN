#!/usr/bin/env python
import datetime
import logging
import functools

import click
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import optimizers
import tensorflow_federated as tff
import kgcn.layers as layers

from datasets.chembl import load_data


def get_logger(level='DEBUG'):
    FORMAT = '%(asctime)-15s - %(pathname)s - %(funcName)s - L%(lineno)3d ::: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger

def build_model_gin(max_n_atoms, max_n_types, protein_max_seqlen, length_one_letter_aa):
    input_adjs = tf.keras.Input(shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(shape=(max_n_atoms, max_n_types), name='features')    
    input_protein_seq = tf.keras.Input(shape=(protein_max_seqlen), name='protein_seq')

    # for graph
    h = layers.GINFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GINFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    
    # for protein sequence
    h_seq = tf.keras.layers.Embedding(length_one_letter_aa, 128, input_length=protein_max_seqlen)(input_protein_seq)
    stride = 4
    #h_seq = tf.keras.layers.GlobalAvePool1D()(h_seq)
    h_seq = tf.keras.layers.Dense(64, activation='relu')(h_seq)
    h_seq = tf.keras.layers.GlobalAveragePooling1D()(h_seq)
    #h_seq = tf.keras.layers.Conv1D(64, stride, padding="same", activation="relu")(h_seq)
    # h_seq = tf.keras.layers.MaxPool1D(stride)(h_seq) 

    # h_seq = tf.keras.layers.LSTM(32, return_sequences=False, go_backwards=True)(h_seq)
    #
    # concat
    h = tf.keras.layers.Concatenate()([h, h_seq])
    #h = tf.keras.layers.Dense(52, activation='relu')(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_protein_seq], outputs=logits)


def build_model_gcn(max_n_atoms, max_n_types, protein_max_seqlen, length_one_letter_aa):
    input_adjs = tf.keras.Input(shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(shape=(max_n_atoms, max_n_types), name='features')    
    input_protein_seq = tf.keras.Input(shape=(protein_max_seqlen), name='protein_seq')

    # for graph
    h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    
    # for protein sequence
    h_seq = tf.keras.layers.Embedding(length_one_letter_aa, 128, input_length=protein_max_seqlen)(input_protein_seq)
    stride = 4
    #h_seq = tf.keras.layers.GlobalAvePool1D()(h_seq)
    h_seq = tf.keras.layers.Dense(64, activation='relu')(h_seq)
    h_seq = tf.keras.layers.GlobalAveragePooling1D()(h_seq)
    #h_seq = tf.keras.layers.Conv1D(64, stride, padding="same", activation="relu")(h_seq)
    # h_seq = tf.keras.layers.MaxPool1D(stride)(h_seq) 

    # h_seq = tf.keras.layers.LSTM(32, return_sequences=False, go_backwards=True)(h_seq)
    #
    # concat
    h = tf.keras.layers.Concatenate()([h, h_seq])
    #h = tf.keras.layers.Dense(52, activation='relu')(h)
    logits = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_protein_seq], outputs=logits)


def client_data(source, n, batch_size, epochs):
    return source.create_tf_dataset_for_client(source.client_ids[n]).repeat(epochs).batch(batch_size)


@click.command()
@click.option('--rounds', default=10, help='the number of updates of the centeral model')
@click.option('--clients', default=4, help='the number of clients')
@click.option('--epochs', default=10, help='the number of training epochs in client traning.')
@click.option('--batchsize', default=32, help='the number of batch size.')
@click.option('--lr', default=0.2, help='learning rate for the central model.')
@click.option('--clientlr', default=0.001, help='learning rate for client models.')
@click.option('--model', default='gcn', help='support gcn or gin.')
@click.option('--ratio', default=None, help='set ratio of the biggest dataset in total datasize.' + \
              ' Other datasets are equally divided. (0, 1)')
def main(rounds, clients, epochs, batchsize, lr, clientlr, model, ratio):
    logger = get_logger()
    subsets = clients + 2
    logger.debug(f'rounds = {rounds}')
    logger.debug(f'clients = {clients}')
    logger.debug(f'subsets = {subsets}')
    logger.debug(f'epochs = {epochs}')
    logger.debug(f'batchsize = {batchsize}')
    logger.debug(f'lr = {lr}')
    logger.debug(f'clientlr = {clientlr}')
    logger.debug(f'model = {model}')
    logger.debug(f'ratio = {ratio}')
    if not model in ['gcn', 'gin']:
        raise Exception(f'not supported model. {model}')
    MAX_N_ATOMS = 150
    MAX_N_TYPES = 120
    PROTEIN_MAX_SEQLEN = 750
    LENGTH_ONE_LETTER_AA = len('XACDEFGHIKLMNPQRSTVWY')

    #subset_ratios = [8/10, 1/10, 1/10]
    #Load simulation data.
    if not ratio is None:
        ratio = float(ratio)
        remains_ratio = [(1 - ratio) / (subsets - 1) for _ in range(subsets - 1)]
        ratios = [ratio, ] + remains_ratio
    else:
        ratios = None
    logger.debug(f'ratios = {ratios}')
    chembl_train = load_data(MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN,
                             subsets, ratios)

    # # Pick a subset of client devices to participate in training.
    all_data = [client_data(chembl_train, n, batchsize, epochs) for n in range(subsets)]
    
    # Wrap a Keras model for use with TFF.
    def _model_fn(model):
        if model == 'gcn':
            model = build_model_gcn(MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN,
                                    LENGTH_ONE_LETTER_AA)
        elif model == 'gin':
            model = build_model_gin(MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN,
                                    LENGTH_ONE_LETTER_AA)
        return tff.learning.from_keras_model(
            model,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
            input_spec=all_data[0].element_spec)
    model_fn = functools.partial(_model_fn, model=model)
    # Simulate a few rounds of training with the selected client devices.
    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(clientlr),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=lr))
    
    evaluation = tff.learning.build_federated_evaluation(model_fn)

    all_test_acc = []
    all_test_loss = []
    all_test_auc = []
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/chembl/' + current_time
    writer = tf.summary.create_file_writer(logdir)
    
    for k in range(subsets):
        state = trainer.initialize()
        test_data_idx = k
        val_data_idx = k - 1 if k == (subsets - 1) else k + 1
        test_data = [all_data[test_data_idx],]
        val_data = [all_data[val_data_idx],]
        train_data = [d for idx, d in enumerate(all_data) if not idx in [test_data_idx, val_data_idx]]
        logger.debug(f'{k} round ->')
        
        for round_num in range(rounds):
            state, metrics = trainer.next(state, train_data)
            train_acc = metrics['train']["binary_accuracy"]
            train_loss = metrics['train']["loss"]
            train_auc = metrics['train']["auc"]
            logger.debug(f'train, round, loss, acc, auc  ===> {round_num:03d}, {train_loss:7.5f}, '
                         f'{train_acc:7.5f}, {train_auc:7.5f},')
            val_metrics = evaluation(state.model, val_data)
            val_acc = val_metrics["binary_accuracy"]
            val_loss = val_metrics["loss"]
            val_auc = val_metrics["auc"]
            logger.debug(f'val, round, loss, acc, auc ===> {round_num:03d}, {val_loss:7.5f}, '
                         f'{val_acc:7.5f}, {val_auc:7.5f},')
            with writer.as_default():
                tf.summary.scalar(f'train_loss{k}', train_loss, step=round_num)
                tf.summary.scalar(f'train_auc{k}', train_auc, step=round_num)
                tf.summary.scalar(f'val_loss{k}', val_loss, step=round_num)
                tf.summary.scalar(f'val_auc{k}', val_auc, step=round_num)                
            
        test_metrics = evaluation(state.model, test_data)
        test_acc = test_metrics["binary_accuracy"]
        test_loss = test_metrics["loss"]
        test_auc = test_metrics["auc"]
        with writer.as_default():        
            tf.summary.scalar(f'test_loss{k}', test_loss, step=round_num)
            tf.summary.scalar(f'test_auc{k}', test_auc, step=round_num)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        all_test_auc.append(test_auc)
        logger.debug(f'test, round, loss, acc, auc ===> {k}, {round_num:03d}, {test_loss:7.5f}, '
                     f' {test_acc:7.5f}, {test_auc:7.5f},')
    logger.debug(f'{clients} >>>> all_test_acc = {all_test_acc}')
    logger.debug(f'{clients} >>>> all_test_loss = {all_test_loss}')
    logger.debug(f'{clients} >>>> all_test_auc = {all_test_auc}')


    
if __name__ == '__main__':
    main()
