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
from sklearn.model_selection import KFold, train_test_split
import optuna

from libs.datasets.chembldb import load_data
from libs.utils import (create_client_data,
                        get_logger)
from libs.platformer import Platformer
from libs.models import build_multimodel_gcn


@click.command()
@click.option('--federated', is_flag=True)
@click.option('--rounds', default=10, help='the number of updates of the centeral model')
@click.option('--clients', default=4, help='the number of clients')
@click.option('--epochs', default=200, help='the number of training epochs in client traning.')
@click.option('--batchsize', default=128, help='the number of batch size.')
@click.option('--lr', default=0.1, help='learning rate for the central model.')
@click.option('--clientlr', default=0.003, help='learning rate for client models.')
@click.option('--model', default='gcn', help='support gcn or gin.')
@click.option('--ratio', default=None, help='set ratio of the biggest dataset in total datasize.' + \
              ' Other datasets are equally divided. (0, 1)')
@click.option('--kfold', default=4, type=int, help='selt number of subsets for the k-fold validation.')
@click.option('--criteria', type=float, default=6., help='set chembl value.')
@click.option('--train_size', type=int, default=5000., help='set training data size.')
@click.option('--datapath', type=str, default='./data/data/exported.db',
              help='set a path of preprocessed database.')
def main(federated, rounds, clients, epochs, batchsize, lr, clientlr, model, ratio, kfold, datapath, train_size, criteria):
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
    logger.debug(f'criteria = {criteria}')            
    logger.debug(f'train_size = {train_size}')            
    dataset = load_data(federated, datapath, MAX_N_ATOMS, MAX_N_TYPES, subsets, ratio, criteria)
    #run_optuna(dataset, kfold, epochs)
    if criteria == 5.:
        class_weight = {0: 2.0, 1: 0.7 }
    elif criteria == 6.:
        class_weight = {0: 0.7, 1: 2.0 }
    elif criteria == 7.:
        class_weight = {0: 0.5, 1: 2.0 }
    logger.info(f'class_weight = {class_weight}')
    if federated:
        federated_learning(dataset, model, rounds, clients,
                           epochs, batchsize, lr, clientlr, ratio, kfold, class_weight, logger)
    else:
        print('normal_learning')
        normal_learning(dataset, epochs, batchsize, clientlr, ratio, kfold, class_weight, train_size, logger)

        
def normal_learning(dataset, epochs, batchsize, clientlr, ratio, kfold, class_weight, train_size, logger):
    MAX_N_ATOMS = 200
    MAX_N_TYPES = 110
    buffer_size = 1000
    train_steps = 10
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # shuffled_dataset = dataset.shuffle(
    #     buffer_size, reshuffle_each_iteration=False)
    shuffled_dataset = dataset
    val_size = train_size // 10
    test_size = 10000
    print('val_size', val_size)
    print('test_size', test_size)
    total_size = 2230981 
    print('{} epochs', total_size / (batchsize * train_steps))
    test_dataset = shuffled_dataset.take(val_size + test_size)
    val = test_dataset.take(val_size).batch(batchsize).prefetch(buffer_size=AUTOTUNE)
    test = test_dataset.skip(test_size).batch(batchsize).prefetch(buffer_size=AUTOTUNE)
    train = shuffled_dataset.skip(val_size + test_size).take(train_size).batch(batchsize).prefetch(buffer_size=AUTOTUNE)

    PROTEIN_MAX_SEQLEN = 1000
    LENGTH_ONE_LETTER_AA = len('XACDEFGHIKLMNPQRSTVWYOUBJZ')
    num_classes = 2

    # for t in train.batch(10):
    #     print(t)
    #     break
    # a
    # pos = 1
    # neg = 10
    # initial_bias = np.log([pos/neg])
    initial_bias = None
    
    model = build_multimodel_gcn(MAX_N_ATOMS, MAX_N_TYPES, num_classes, PROTEIN_MAX_SEQLEN,
                                 LENGTH_ONE_LETTER_AA, output_bias=initial_bias)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            clientlr,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True)

    loss = tf.keras.losses.BinaryCrossentropy()
    loss = tfa.losses.SigmoidFocalCrossEntropy()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
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
    checkpoint_path = f"models/chembl.ckpt.{os.getpid()}"
    logger.info(f'checkpoint_path {checkpoint_path}')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     monitor='val_auc',
                                                     mode='max',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)
    metric = model.fit(train, epochs=epochs,
                       validation_data=val,
                       #steps_per_epoch=train_steps,
                       callbacks=[cp_callback, callback],
                       class_weight=class_weight,
                       verbose=0)
    metric = model.evaluate(test, verbose=2)
    print(metric)
    logger.info(f'final results => {metric}')
    kappa = kfold_cv
    
def kfold_cv(nfold, dataset, epochs, loss_type, lr, optim,
             batchsize, gcn_hiddens, gcn_layers, linear_hiddens):
    dataset = dataset.shuffle(100000)
    
    kf = KFold(n_splits=nfold)
    return 0.1

    
def federated_learning(dataset, model, rounds, clients,
                       epochs, batchsize, lr, clientlr, ratio, kfold, class_weight, logger):
    if not model in ['gcn', 'gin']:
        raise Exception(f'not supported model. {model}')
    MAX_N_ATOMS = 200
    MAX_N_TYPES = 110
    CLIENT_SIZE = 300
    PROTEIN_MAX_SEQLEN = 750
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    LENGTH_ONE_LETTER_AA = len('XACDEFGHIKLMNPQRSTVWYOUBJZ')
    num_classes = 1
    pos = 1
    neg = 10
    # initial_bias = np.log([pos/neg])
    # Pick a subset of client devices to participate in training.
    initial_bias = None
    all_data = [dataset.take(CLIENT_SIZE).repeat(epochs).batch(batchsize).prefetch(buffer_size=AUTOTUNE) for _ in range(clients)]

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
    metric_names = ['auc']
    loss_name = 'loss'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', 'chembldb', current_time)
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
        print('test_metrics', test_metrics)
        with writer.as_default():        
            tf.summary.scalar(f'test_loss_name{k}', test_loss, step=round_num)
            tf.summary.scalar(f'test_{metric_names[0]}{k}', test_acc, step=round_num)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        logger.debug(f'test, round, loss, acc ===> {k}, {round_num:03d}, {test_loss:7.5f}, '
                     f' {test_acc:7.5f}')
        break

    logger.debug(f'{clients} >>>> all_test_acc = {all_test_acc}')
    logger.debug(f'{clients} >>>> all_test_loss = {all_test_loss}')


def _objective(trial, nfold, dataset, epochs):
    tf.compat.v1.reset_default_graph()
    MAX_N_ATOMS = 150
    MAX_N_TYPES = 100

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)        
    optim = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    batchsize = trial.suggest_int("batch_size", 8, 256, step=64)
    loss_type = trial.suggest_categorical("_loss_type", ['binary', 'focal'])
    num_gcn_layers = trial.suggest_int("num_gcn_layers", 1, 3)
    num_linear_layers = trial.suggest_int("num_linear_layers", 1, 3)
    gcn_hiddens = []
    gcn_layers = []    
    linear_hiddens = []
    _gcn_layers = ['APPNPConv', 'ARMAConv', 'ChebConv',
                   'DiffusionConv', 'GATConv', 'GCNConv', 'GCSConv']
    for i in range(num_gcn_layers):
        gcn_hiddens.append(trial.suggest_int(f"gcn_hiddens_{i}", 16, 128, step=32))
        gcn_layers.append(trial.suggest_categorical(f"gcn_layers_{i}", _gcn_layers))
    for i in range(num_linear_layers):    
        linear_hiddens.append(trial.suggest_int(f"linear_hiddens_{i}", 8, 128, step=32))

    kappa = kfold_cv(nfold, dataset, epochs, loss_type, lr, optim,
                     batchsize, gcn_hiddens, gcn_layers, linear_hiddens)
    return kappa

def run_optuna(dataset, nfold, epochs):
    study_name = 'ChEMBLDB'
    objective = functools.partial(_objective, nfold=nfold, dataset=dataset, epochs=epochs)
    study = optuna.create_study(study_name=study_name,
                                storage=f'sqlite:///./chembl_study_{nfold}_fold.db',
                                load_if_exists=True,
                                direction='maximize')
    study.optimize(objective, n_trials=100)
    #print(_objective(study.best_trial, datapath=datapath, task_name=task_name, epochs=epochs))
    print(study.best_trial)
    print(study.best_params)
    
    
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
