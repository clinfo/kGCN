#!/usr/bin/env python
import datetime
import functools

import click
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import optimizers
import tensorflow_federated as tff
import kgcn.layers as layers

from libs.datasets.adme import load_data
from libs.utils import (create_client_data,
                        get_logger)
from libs.platformer import Platformer


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
@click.option('--kfold', is_flag=True, help='turn on k-fold validation.')
def main(rounds, clients, epochs, batchsize, lr, clientlr, model, ratio, kfold):
    logger = get_logger("ADME")
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


    # Load simulation data.
    if not ratio is None:
        ratio = float(ratio)
        remains_ratio = [(1 - ratio) / (subsets - 1) for _ in range(subsets - 1)]
        ratios = [ratio, ] + remains_ratio
    else:
        ratios = None
    logger.debug(f'ratios = {ratios}')
    
    chembl_train = load_data(MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN, subsets, ratios, loaddir)

    # Pick a subset of client devices to participate in training.
    all_data = [create_client_data(chembl_train, n, batchsize, epochs) for n in range(subsets)]
    
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
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/adme/' + current_time
    
    platformer = Platformer(trainer, model_fn, all_data, logdir=logdir)
    platformer.run()
    
    
    # for k in range(subsets):
    #     state = trainer.initialize()
    #     test_data_idx = k
    #     val_data_idx = k - 1 if k == (subsets - 1) else k + 1
    #     test_data = [all_data[test_data_idx],]
    #     val_data = [all_data[val_data_idx],]
    #     train_data = [d for idx, d in enumerate(all_data) if not idx in [test_data_idx, val_data_idx]]
    #     logger.debug(f'{k} round ->')
        
    #     for round_num in range(rounds):
    #         state, metrics = trainer.next(state, train_data)
    #         train_acc = metrics['train']["binary_accuracy"]
    #         train_loss = metrics['train']["loss"]
    #         train_auc = metrics['train']["auc"]
    #         logger.debug(f'train, round, loss, acc, auc  ===> {round_num:03d}, {train_loss:7.5f}, '
    #                      f'{train_acc:7.5f}, {train_auc:7.5f},')
    #         val_metrics = evaluation(state.model, val_data)
    #         val_acc = val_metrics["binary_accuracy"]
    #         val_loss = val_metrics["loss"]
    #         val_auc = val_metrics["auc"]
    #         logger.debug(f'val, round, loss, acc, auc ===> {round_num:03d}, {val_loss:7.5f}, '
    #                      f'{val_acc:7.5f}, {val_auc:7.5f},')
    #         with writer.as_default():
    #             tf.summary.scalar(f'train_loss{k}', train_loss, step=round_num)
    #             tf.summary.scalar(f'train_auc{k}', train_auc, step=round_num)
    #             tf.summary.scalar(f'val_loss{k}', val_loss, step=round_num)
    #             tf.summary.scalar(f'val_auc{k}', val_auc, step=round_num)                
            
    #     test_metrics = evaluation(state.model, test_data)
    #     test_acc = test_metrics["binary_accuracy"]
    #     test_loss = test_metrics["loss"]
    #     test_auc = test_metrics["auc"]
    #     with writer.as_default():        
    #         tf.summary.scalar(f'test_loss{k}', test_loss, step=round_num)
    #         tf.summary.scalar(f'test_auc{k}', test_auc, step=round_num)
    #     all_test_acc.append(test_acc)
    #     all_test_loss.append(test_loss)
    #     all_test_auc.append(test_auc)
    #     logger.debug(f'test, round, loss, acc, auc ===> {k}, {round_num:03d}, {test_loss:7.5f}, '
    #                  f' {test_acc:7.5f}, {test_auc:7.5f},')
    # logger.debug(f'{clients} >>>> all_test_acc = {all_test_acc}')
    # logger.debug(f'{clients} >>>> all_test_loss = {all_test_loss}')
    # logger.debug(f'{clients} >>>> all_test_auc = {all_test_auc}')


    
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
