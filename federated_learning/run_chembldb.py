#!/usr/bin/env python
import os
import datetime
import functools
import collections

import click
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_federated as tff
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
import kgcn.layers as layers

from libs.datasets.chembldb import load_data
from libs.utils import (create_client_data,
                        get_logger)
from libs.platformer import Platformer
from libs.models import build_multimodel_gcn


@click.command()
@click.option('--federated', is_flag=True)
@click.option('--rounds', default=10, help='the number of updates of the centeral model')
@click.option('--clients', default=4, help='the number of clients')
@click.option('--epochs', default=10, help='the number of training epochs in client traning.')
@click.option('--batchsize', default=256, help='the number of batch size.')
@click.option('--lr', default=0.1, help='learning rate for the central model.')
@click.option('--clientlr', default=0.001, help='learning rate for client models.')
@click.option('--model', default='gcn', help='support gcn or gin.')
@click.option('--ratio', default=None, help='set ratio of the biggest dataset in total datasize.' + \
              ' Other datasets are equally divided. (0, 1)')
@click.option('--kfold', is_flag=True, help='turn on k-fold validation.')
@click.option('--datapath', type=str, default='./data/data/exported.db',
              help='set a path of preprocessed database.')
def main(federated, rounds, clients, epochs, batchsize, lr, clientlr, model, ratio, kfold, datapath):
    logger = get_logger("ChEBMLDB")
    subsets = clients + 2
    MAX_N_ATOMS = 150
    MAX_N_TYPES = 100
    
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
    logger.debug(f'datapath = {datapath}')        
    dataset = load_data(federated, datapath, MAX_N_ATOMS, MAX_N_TYPES, subsets, ratio)
    
    if federated:
        federated_learning(dataset, model, rounds, clients,
                           epochs, batchsize, lr, clientlr, ratio, kfold, logger)
    else:
        normal_learning(dataset, epochs, batchsize, clientlr, model, ratio, kfold, logger)

        
def normal_learning(dataset, epochs, batchsize, clientlr, model, ratio, kfold, logger):
    MAX_N_ATOMS = 200
    MAX_N_TYPES = 110
    buffer_size = 10000
    train_steps = 200
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # shuffled_dataset = dataset.shuffle(
    #     buffer_size, reshuffle_each_iteration=False)
    shuffled_dataset = dataset
    
    val_size = train_steps // 10 * batchsize
    test_size = train_steps // 10 * batchsize
    test_dataset = shuffled_dataset.take(val_size + test_size)
    val = test_dataset.take(val_size).batch(batchsize).prefetch(buffer_size=AUTOTUNE)
    test = test_dataset.skip(test_size).batch(batchsize).prefetch(buffer_size=AUTOTUNE)
    train = shuffled_dataset.skip(val_size + test_size).batch(batchsize).prefetch(buffer_size=AUTOTUNE)

    PROTEIN_MAX_SEQLEN = 750
    LENGTH_ONE_LETTER_AA = len('XACDEFGHIKLMNPQRSTVWYOUBJZ')
    num_classes = 2

    pos = 1
    neg = 10
    initial_bias = np.log([pos/neg])
    
    model = build_multimodel_gcn(MAX_N_ATOMS, MAX_N_TYPES, num_classes, PROTEIN_MAX_SEQLEN,
                                 LENGTH_ONE_LETTER_AA, output_bias=initial_bias)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=clientlr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC(), tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.FalseNegatives(),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives()])

    model.summary()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                verbose=1,
                                                patience=10,
                                                mode='max',
                                                restore_best_weights=True)
    checkpoint_path = "models/chembl.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # チェックポイントコールバックを作る
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     monitor='val_auc',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    #lr_callback = tfa.optimizers.TriangularCyclicalLearningRate(1e-3, 0.1, 40)
    import math
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 - (0.0099 * epoch)/ epochs, verbose=True)
    metric = model.fit(train, epochs=epochs,
                       validation_data=val,
                       steps_per_epoch=train_steps,
                       callbacks=[lr_decay, cp_callback],
                       class_weight = {0: 1.2, 1: 0.7 })
    metric = model.evaluate(test, steps=10)
    print(metric)

        
def federated_learning(dataset, model, rounds, clients,
                       epochs, batchsize, lr, clientlr, ratio, kfold, logger):
    if not model in ['gcn', 'gin']:
        raise Exception(f'not supported model. {model}')
    MAX_N_ATOMS = 200
    MAX_N_TYPES = 110
    CLIENT_SIZE = 1000
    PROTEIN_MAX_SEQLEN = 750
    LENGTH_ONE_LETTER_AA = len('XACDEFGHIKLMNPQRSTVWYOUBJZ')
    num_classes = 1
    pos = 1
    neg = 10
    # initial_bias = np.log([pos/neg])
    # Pick a subset of client devices to participate in training.

    all_data = [dataset.take(CLIENT_SIZE).repeat(epochs).batch(batchsize) for _ in range(clients)]

    def model_fn():
        #if model == 'gcn':
        model = build_multimodel_gcn(MAX_N_ATOMS, MAX_N_TYPES, num_classes, PROTEIN_MAX_SEQLEN,
                                     LENGTH_ONE_LETTER_AA, output_bias=initial_bias)
        
        return tff.learning.from_keras_model(
            model,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(),
                     tf.keras.metrics.AUC(), tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.TruePositives(), tf.keras.metrics.FalseNegatives(),
                     tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives()],
            input_spec=all_data[0].element_spec)
    
    # Simulate a few rounds of training with the selected client devices.
    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(clientlr),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=lr))
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    
    all_test_acc = []
    all_test_loss = []
    all_test_auc = []
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', 'toxicity', current_time)
    writer = tf.summary.create_file_writer(logdir)
    
    for k in range(clients):
        state = trainer.initialize()
        test_data_idx = k
        val_data_idx = k - 1 if k == (clients - 1) else k + 1
        test_data = [all_data[test_data_idx],]
        val_data = [all_data[val_data_idx],]
        train_data = [d for idx, d in enumerate(all_data) if not idx in [test_data_idx, val_data_idx]]
        logger.debug(f'{k} round ->')
        
        for round_num in range(rounds):
            state, metrics = trainer.next(state, train_data)
            print(metrics['train'])            
            train_acc = metrics['train'][metric_names[0]]
            train_loss = metrics['train'][loss_name]
            logger.debug(f'train, round, loss, acc  ===> {round_num:03d}, {train_loss:7.5f}, '
                         f'{train_acc:7.5f},')
            val_metrics = evaluation(state.model, val_data)
            val_acc = val_metrics[metric_names[0]]
            val_loss = val_metrics[loss_name]
            logger.debug(f'val, round, loss, acc ===> {round_num:03d}, {val_loss:7.5f}, '
                         f'{val_acc:7.5f},')
            with writer.as_default():
                tf.summary.scalar(f'train_loss{k}', train_loss, step=round_num)
                tf.summary.scalar(f'train_acc{k}', train_acc, step=round_num)
                tf.summary.scalar(f'val_loss{k}', val_loss, step=round_num)
                tf.summary.scalar(f'val_acc{k}', val_acc, step=round_num)                
            
        test_metrics = evaluation(state.model, test_data)
        test_loss = test_metrics[loss_name]        
        test_acc = test_metrics[metric_names[0]]
        with writer.as_default():        
            tf.summary.scalar(f'test_loss_name{k}', test_loss, step=round_num)
            tf.summary.scalar(f'test_{metric_names[0]}{k}', test_acc, step=round_num)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        logger.debug(f'test, round, loss, acc ===> {k}, {round_num:03d}, {test_loss:7.5f}, '
                     f' {test_acc:7.5f}')
    logger.debug(f'{clients} >>>> all_test_acc = {all_test_acc}')
    logger.debug(f'{clients} >>>> all_test_loss = {all_test_loss}')

    
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
