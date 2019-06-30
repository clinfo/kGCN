#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for GCN training for molecule profiling.

This module trains a Graph Convolutional Network for predictions of molecular properties.

USAGE
-----
1. Prepare .tfrecords files in a dataset folder.

The files that are named '*[train, eval, test]*.tfrecords' are used for training, eval, test.

You can have multiple files for training, etc.Alternatively, you can just have one file that contains multiple examples for training.

The format of serialized data in .tfrecords:

features = {
        'label': tf.FixedLenFeature([label_length], tf.float32),
        'mask_label': tf.FixedLenFeature([label_length], tf.float32),
        'adj_row': tf.VarLenFeature(tf.int64),
        'adj_column': tf.VarLenFeature(tf.int64),
        'adj_values': tf.VarLenFeature(tf.float32),
        'adj_elem_len': tf.FixedLenFeature([1], tf.int64),
        'adj_degrees': tf.VarLenFeature(tf.int64),
        'feature_row': tf.VarLenFeature(tf.int64),
        'feature_column': tf.VarLenFeature(tf.int64),
        'feature_values': tf.VarLenFeature(tf.float32),
        'feature_elem_len': tf.FixedLenFeature([1], tf.int64),
        'size': tf.FixedLenFeature([2], tf.int64)
}

2. python task_sparse_gcn.py --dataset your_dataset --other_flags

@author: taro.kiritani
"""
import ast
from distutils.version import StrictVersion
import json
import os
import random
import re
import shutil
import tempfile

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO, get_matching_files, delete_file

tf.logging.set_verbosity(tf.logging.INFO)

try:
    from .model_functions import gcn_multitask_model_sparse
except ModuleNotFoundError:
    from model_functions import gcn_multitask_model_sparse


def make_parse_fn(example_proto, label_length):
    features = {
        "label": tf.FixedLenFeature([label_length], tf.int64),
        "mask_label": tf.FixedLenFeature([label_length], tf.int64),
        "adj_row": tf.VarLenFeature(tf.int64),
        "adj_column": tf.VarLenFeature(tf.int64),
        "adj_values": tf.VarLenFeature(tf.float32),
        "adj_elem_len": tf.FixedLenFeature([1], tf.int64),
        "adj_degrees": tf.VarLenFeature(tf.int64),
        "feature_row": tf.VarLenFeature(tf.int64),
        "feature_column": tf.VarLenFeature(tf.int64),
        "feature_values": tf.VarLenFeature(tf.float32),
        "feature_elem_len": tf.FixedLenFeature([1], tf.int64),
        "size": tf.FixedLenFeature([2], tf.int64),
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features, parsed_features["label"]


def make_input_fn(dataset):
    def input_fn():
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return input_fn


def main(args):
    FLAGS = tf.app.flags.FLAGS
    gcn_dims = ast.literal_eval(FLAGS.gcn_dims)
    if not all([isinstance(d, int) for d in gcn_dims]):
        raise ValueError("gcn_dims should be a list of integers.")

    # convention: dataset_folder/*_[fold_num]_['train' or 'eval' or 'test'].tfrecords
    dataset_folder = FLAGS.dataset
    with FileIO(get_matching_files(os.path.join(dataset_folder, "tasks.txt"))[0], "r") as text_file:
        task_names = text_file.readlines()
    task_num = len(task_names)
    if FLAGS.how_to_split == "tet":
        files_train = os.path.join(dataset_folder, "*train*.tfrecords")
        files_eval = os.path.join(dataset_folder, "*eval*.tfrecords")
        files_test = os.path.join(dataset_folder, "*test*.tfrecords")
        folds = [
            [
                get_matching_files(files_train),
                get_matching_files(files_eval),
                get_matching_files(files_test),
            ]
        ]
    elif FLAGS.how_to_split == "kfold":
        all_fold_strs = [str(k) for k in range(FLAGS.fold_num)]
        folds = []
        for fold_num in range(FLAGS.fold_num):
            files_eval = os.path.join(
                dataset_folder, "*_" + str(fold_num) + "_*.tfrecords"
            )
            files_test = os.path.join(
                dataset_folder, "*_" + str(fold_num) + "_*.tfrecords"
            )
            train_folds = [fold for fold in all_fold_strs if fold != fold_num]
            train_folds_str = "[" + ",".join(train_folds) + "]"
            files_train = os.path.join(
                dataset_folder, "*_" + train_folds_str + "_*.tfrecords"
            )
            folds.append(
                [
                    get_matching_files(files_train),
                    get_matching_files(files_eval),
                    get_matching_files(files_test),
                ]
            )
    elif FLAGS.how_to_split == "811":
        files = os.path.join(dataset_folder, "*.tfrecords")
        files = get_matching_files(files)
        random.shuffle(files)
        files_train = files[: int(len(files) * 0.8)]
        files_eval = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        files_test = files[int(len(files) * 0.9) :]
        folds = [[files_train, files_eval, files_test]]
    parser = lambda example_proto: make_parse_fn(example_proto, task_num)
    tf.logging.info(FLAGS["job-dir"].value)
    for fold_num, fold in enumerate(folds):
        if len(folds) > 1:
            model_dir = FLAGS["job-dir"].value + "_fold_" + str(fold_num)
        else:
            model_dir = FLAGS["job-dir"].value
            count_examples = len(fold[0])
        steps_per_epoch = count_examples // FLAGS.batch_size
        tf.logging.info(
            "example num: {}, steps per epoch: {}".format(
                count_examples, steps_per_epoch
            )
        )
        count_examples_eval = len(fold[1])
        steps_per_epoch_eval = count_examples_eval // FLAGS.batch_size
        tf.logging.info(
            "example num: {}, steps per epoch: {}".format(
                count_examples_eval, steps_per_epoch_eval
            )
        )
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            record = next(tf.python_io.tf_record_iterator(fold[0][0]))
            input_dim = parser(record)[0]["size"][1].eval()
        train_eval_test_datasets = []
        tempds = []
        for k, files in enumerate(fold):
            if k == 0:
                epoch_num = FLAGS.epochs
            else:
                epoch_num = 1
            dataset = tf.data.TFRecordDataset(files, )
            dataset = dataset.map(parser)
            if FLAGS.cache == "memory":
                dataset.cache()
            elif FLAGS.cache == "temp_file":
                tempd = tempfile.mkdtemp()
                tempds.append(tempd)
                dataset.cache(tempd)
            if k == 0:
                if FLAGS.shuffle_on_memory == -1:
                    dataset = dataset.shuffle(
                        count_examples, reshuffle_each_iteration=True
                    )
                elif FLAGS.shuffle_on_memory > 0:
                    dataset = dataset.shuffle(
                        FLAGS.shuffle_on_ememory, reshuffle_each_iteration=True
                    )
            dataset = dataset.batch(FLAGS.batch_size)
            dataset = dataset.prefetch(buffer_size=1)
            if StrictVersion(tf.__version__) > StrictVersion("1.9.0"):
                dataset = dataset.repeat(epoch_num)
                throttle_secs = 1
            else:
                dataset = dataset.repeat(1)
                throttle_secs = 6000000
            train_eval_test_datasets.append(dataset)
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        estimator_config = tf.estimator.RunConfig(
            session_config=session_config,
            save_checkpoints_steps=steps_per_epoch,
            keep_checkpoint_max=0,
        )
        gcn_classifier = tf.estimator.Estimator(
            model_fn=gcn_multitask_model_sparse,
            model_dir=model_dir,
            params={
                "learning_rate": FLAGS.learning_rate,
                "task_names": task_names,
                "do_rate": FLAGS.dropout,
                "task_num": task_num,
                "out_dims": gcn_dims,
                "dtype": tf.float32,
                "use_bias": True,
                "dense_dim": FLAGS.dense_node_num,
                "mltask": FLAGS.mltask,
                "input_dim": input_dim,
                "batch_normalize": FLAGS.batch_normalize,
                "max_pool": FLAGS.max_pool,
                "max_degree": FLAGS.max_degree,
                "normalize": FLAGS.normalize_adj,
                "multitask": FLAGS.multitask,
                "num_classes": FLAGS.num_classes
            },
            config=estimator_config,
        )
        tf.logging.info(FLAGS.epochs)
        train_spec = tf.estimator.TrainSpec(
            input_fn=make_input_fn(train_eval_test_datasets[0]),
            max_steps=(steps_per_epoch * FLAGS.epochs),
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=make_input_fn(train_eval_test_datasets[1]),
            steps=steps_per_epoch_eval,
            throttle_secs=throttle_secs,
        )
        tf.estimator.train_and_evaluate(gcn_classifier, train_spec, eval_spec)
        ea = event_accumulator.EventAccumulator(os.path.join(model_dir, "eval"))
        ea.Reload()
        keys = ea.scalars.Keys()
        loss = [scalar.value for scalar in ea.Scalars("loss")]
        smallest_index = loss.index(min(loss))
        smallest_step = ea.Scalars("loss")[smallest_index].step
        if FLAGS.run_test:
            test_result = gcn_classifier.evaluate(
                input_fn=make_input_fn(train_eval_test_datasets[2]),
                steps=steps_per_epoch_eval,
                checkpoint_path=os.path.join(
                    model_dir, "model.ckpt-{}".format(smallest_step)
                ),
            )
            try:
                os.mkdir(os.path.join(model_dir, "test"))
            except FileExistsError:
                pass

            test_result = {k: np.float(v) for k, v in test_result.items()}
            with open(os.path.join(model_dir, "test", "test.json"), "w") as outfile:
                json.dump(test_result, outfile)
        recorded_models = get_matching_files(
            os.path.join(model_dir, "model.ckpt-*.index")
        )
        models_to_be_deleted = [
            re.search("model.ckpt-\d+", model).group(0)[11:]
            for model in recorded_models
        ]
        models_to_be_deleted.remove("0")
        models_to_be_deleted.remove(str(smallest_step))
        if len(models_to_be_deleted) > 0:
            max_step = max([int(num) for num in models_to_be_deleted])
            models_to_be_deleted.remove(str(max_step))
            for model in models_to_be_deleted:
                delete_this = get_matching_files(
                    os.path.join(model_dir, "model.ckpt-" + model + ".*")
                )
                for f in delete_this:
                    delete_file(f)
        feature_spec = {
            "label": tf.FixedLenFeature([task_num], tf.float32),
            "mask_label": tf.FixedLenFeature([task_num], tf.float32),
            "adj_row": tf.VarLenFeature(tf.int64),
            "adj_column": tf.VarLenFeature(tf.int64),
            "adj_values": tf.VarLenFeature(tf.float32),
            "adj_elem_len": tf.FixedLenFeature([1], tf.int64),
            "adj_degrees": tf.VarLenFeature(tf.int64),
            "feature_row": tf.VarLenFeature(tf.int64),
            "feature_column": tf.VarLenFeature(tf.int64),
            "feature_values": tf.VarLenFeature(tf.float32),
            "feature_elem_len": tf.FixedLenFeature([1], tf.int64),
            "size": tf.FixedLenFeature([2], tf.int64),
        }

        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec
        )
        gcn_classifier.export_savedmodel(
            os.path.join(model_dir, "export_dir"),
            serving_input_receiver_fn,
            checkpoint_path=os.path.join(model_dir, "model.ckpt-" + str(smallest_step)),
            as_text=True,
        )
        for tmpd in tempds:
            shutil.rmtree(tmpd)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        "job-dir", "./train", "directory path for checkpoints and training results."
    )
    tf.app.flags.DEFINE_float("learning_rate", 0.003, "set learning rate.")
    tf.app.flags.DEFINE_float("dropout", 0.2, "set dropout rate.")
    tf.app.flags.DEFINE_integer("batch_size", 128, "set batch size.")
    tf.app.flags.DEFINE_integer("epochs", 200, "set the number of epochs.")
    tf.app.flags.DEFINE_integer(
        "dense_node_num", 128, "set the number of nodes in last dense layer."
    )
    tf.app.flags.DEFINE_string(
        "mltask",
        "classification",
        "set a type of learning task. [classfication, regression]",
    )
    tf.app.flags.DEFINE_string(
        "dataset", "solubility", "choose a directory containing .tfrecords files."
    )
    tf.app.flags.DEFINE_string(
        "gcn_dims",
        "[64, 64]",
        "List of dimensions in each gcn layer. The number of numbers determine the number of layers. Use list espression in  quotation marks.",
    )
    tf.app.flags.DEFINE_string("how_to_split", "tet", "either '811', 'tet' or 'kfold'")
    tf.app.flags.DEFINE_integer(
        "fold_num", 5, "fold number, only used with --how_to_split kfold"
    )
    tf.app.flags.DEFINE_integer(
        "shuffle_on_memory",
        0,
        "the buffer size for shuffling of train data. 0 for no shuffling. -1 for perfect shuffling. Note that file names are shuffled anyway.",
    )
    tf.app.flags.DEFINE_string("cache", None, "'memory' or 'temp_file', or undefined")
    tf.app.flags.DEFINE_boolean(
        "run_test",
        True,
        "if true, run prediction after training, with the model with the lowest eval loss.",
    )
    tf.app.flags.DEFINE_boolean(
        "batch_normalize", False, "if true, use graph batch normalization."
    )
    tf.app.flags.DEFINE_boolean("max_pool", False, "if true, use graph max pooling.")
    tf.app.flags.DEFINE_integer(
        "max_degree",
        0,
        "adjacency matrices are split into max_degree matrices, based on the degrees of nodes. Assumes the matrices are already split into channels by the preprocessing program. The nodes with the degree larger than this number will be collected in a single matrix. Degree is the number of edges connected to a node, excluding the one to itself.",
    )
    tf.app.flags.DEFINE_boolean("normalize_adj", False, "if true, adjacency matrix is normalized.")
    tf.app.flags.DEFINE_boolean("multitask", True, "True for binary classification of multi-tasks, False for multi-class classification of a single task")
    tf.app.flags.DEFINE_integer("num_classes", 2, "number of classes for multi-class classificatoin.")
    tf.app.run()
