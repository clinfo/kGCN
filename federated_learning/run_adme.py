#!/usr/bin/env python
import os
import datetime
import functools

import click
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
import kgcn.layers as layers

from libs.datasets.adme import load_data
from libs.utils import (create_client_data,
                        get_logger)
from libs.platformer import Platformer
from libs.models import build_model_gin, build_model_gcn


@click.command()
@click.option('--federated', is_flag=True)
@click.option('--rounds', default=10, help='the number of updates of the centeral model')
@click.option('--clients', default=4, help='the number of clients')
@click.option('--epochs', default=10, help='the number of training epochs in client traning.')
@click.option('--batchsize', default=32, help='the number of batch size.')
@click.option('--lr', default=0.1, help='learning rate for the central model.')
@click.option('--clientlr', default=0.001, help='learning rate for client models.')
@click.option('--model', default='gcn', help='support gcn or gin.')
@click.option('--ratio', default=None, help='set ratio of the biggest dataset in total datasize.' + \
              ' Other datasets are equally divided. (0, 1)')
@click.option('--kfold', is_flag=True, help='turn on k-fold validation.')
@click.option('--datapath', type=str, default='./data/ADME_Activity and SDFs/Activity and SDFs',
              help='set a data directory path which contains activity.xml and sdf-files.')
@click.option('--task_name', type=click.Choice(['abso_sol_cls', 'abso_cacco_cls', 'abso_cacco_reg', 'abso_llc_reg',
                                                'dist_llc_cls', 'dist_fu_man_reg', 'dist_fu_hum_reg', 'dist_fu_rat_reg',
                                                'dist_rb_rat_reg', 'meta_clint_reg', 'excr_fe_cls']),
              default='abso_sol_cls', help='set task name. ???_+++_***:: ???_+++ is dataset name, *** is task(cls=classification, reg=regression).')
def main(federated, rounds, clients, epochs, batchsize, lr, clientlr, model, ratio, kfold, datapath, task_name):
    logger = get_logger("ADME")
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
    logger.debug(f'task_name = {task_name}')    

    dataset = load_data(federated, datapath, task_name, MAX_N_ATOMS, MAX_N_TYPES, subsets, ratio)
    
    if federated:
        federated_learning(task_name, dataset, model, rounds, clients,
                           epochs, batchsize, lr, clientlr, ratio, kfold, logger)
    else:
        normal_learning(task_name, dataset, epochs, batchsize, clinetlr, model, ratio, kfold, logger)

        
def normal_learning(task_name, dataset, epochs, batchsize, lr, model, ratio, kfold):
    MAX_N_ATOMS = 150
    MAX_N_TYPES = 100
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
    task_type = task_name.split('_')[-1]
    task_target = task_name.split('_')[-2]    
    if task_type == 'cls':
        num_classes = 2            
        if task_target == 'llc':
            num_classes = 3
    else:
        # regression
        num_classes = 1        
    
    if model == 'gcn':
        model = build_model_gcn(MAX_N_ATOMS, MAX_N_TYPES, num_classes)
    elif model == 'gin':
        model = build_model_gin(MAX_N_ATOMS, MAX_N_TYPES, num_classes)

    if task_type == 'cls':
        print(task_name)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               tf.keras.metrics.AUC()])
        # 
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        #               loss=tf.keras.losses.BinaryCrossentropy(),
        #               metrics=[tf.keras.metrics.BinaryAccuracy(),
        #                        tf.keras.metrics.AUC(),
        #                        tfa.metrics.CohenKappa(num_classes=2)])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanSquaredError()])
    
    model.summary()
    model.fit(train.shuffle(100).batch(batchsize), epochs=epochs)
    model.evaluate(test.batch(1))
    prediction = model.predict(test.batch(1))
    
    if task_type == 'cls':
        predictions = tf.math.argmax(prediction, axis=1)
        trues = tf.math.argmax([t[-1] for t in test], axis=1)
        kappa_metrics = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
        kappa_metrics.update_state(trues, predictions)
        print('predictions\n', predictions)
        print('trues\n', trues)        
        print('kappa', kappa_metrics.result())
    else:
        trues = [t[-1] for t in test]
        predictions = prediction[:, -1]
        r2_metrics = tfa.metrics.r_square.RSquare()
        r2_metrics.update_state(trues, predictions)        
        print('predictions\n', predictions)
        print('trues\n', np.array(trues))
        print('r2', r2_metrics.result().numpy)

        
def federated_learning(task_name, dataset, model, rounds, clients,
                       epochs, batchsize, lr, clientlr, ratio, kfold, logger):
    if not model in ['gcn', 'gin']:
        raise Exception(f'not supported model. {model}')
    MAX_N_ATOMS = 150
    MAX_N_TYPES = 100

    # Pick a subset of client devices to participate in training.
    all_data = [create_client_data(dataset, n, batchsize, epochs) for n in range(clients)]
    task_type = task_name.split('_')[-1]
    task_target = task_name.split('_')[-2]    
    
    # Wrap a Keras model for use with TFF.
    if task_type == 'cls':
        num_classes = 2            
        if task_target == 'llc':
            num_classes = 3
    else:
        # regression
        num_classes = 1        
    def _model_fn(model, task_type, num_classes):
        if model == 'gcn':
            model = build_model_gcn(MAX_N_ATOMS, MAX_N_TYPES, num_classes)
        elif model == 'gin':
            model = build_model_gin(MAX_N_ATOMS, MAX_N_TYPES, num_classes)
        if task_type == 'cls':            
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
        else:
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.MeanSquaredError(),]
        return tff.learning.from_keras_model(
            model, loss=loss, metrics=metrics, input_spec=all_data[0].element_spec)
    if task_type == 'cls':            
        loss_name = 'loss'
        metric_names = ['categorical_accuracy', 'auc']
    else:
        loss = 'loss'
        metric_names = ['meansquarederror',]
    
    model_fn = functools.partial(_model_fn, model=model,
                                 task_type=task_type, num_classes=num_classes)
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
