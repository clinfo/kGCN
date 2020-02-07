import argparse
import importlib
import json
import math
import os
import sys
import time
from typing import Iterable, List

import numpy as np
import tensorflow as tf


tf.enable_eager_execution()  # only to count num of elements in datasets
tf.logging.set_verbosity(tf.logging.INFO)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


def get_default_config():
    config = {}
    config["model.py"] = "model"
    config["dataset"] = "data.jbl"
    config["validation_dataset"] = None
    # optimization parameters
    config["epoch"] = 50
    config["batch_size"] = 10
    config["patience"] = 0
    config["learning_rate"] = 0.3
    config["validation_data_rate"] = 0.3
    config["shuffle_data"] = False
    config["k-fold_num"] = 2
    # model parameters
    config["with_feature"] = True
    config["with_node_embedding"] = False
    config["embedding_dim"] = 10
    config["normalize_adj_flag"] = False
    config["split_adj_flag"] = False
    config["order"] = 1
    config["param"] = None
    # model
    config["save_interval"] = 10
    config["save_model_path"] = "model"
    # result/info
    #config["save_result_train"]=None
    config["save_result_valid"] = None
    config["save_result_test"] = None
    config["save_result_cv"] = None
    config["save_info_train"] = None
    config["save_info_valid"] = None
    config["save_info_test"] = None
    config["save_info_cv"] = None
    config["make_plot"] = False
    config["plot_path"] = "./result/"
    config["plot_multitask"] = False
    config["task"] = "classification"
    config["retrain"] = None
    #
    config["profile"] = False
    config["export_model"] = None
    config["stratified_kfold"] = False

    return config


def make_parse_fn(example_proto, feature_spec):
    """Parses example proto
    Args:
        example_proto
        feature_spec
    """
    parsed_features = tf.io.parse_single_example(example_proto, feature_spec)
    label = parsed_features.pop("label")
    return parsed_features, label


def make_input_fn(files, input_parser, cache, shuffle_on_memory, epoch_num, split=None, take_these_splits=None):
    def input_fn():
        dataset = collect_data(files, input_parser, split, take_these_splits)

        if cache:
            dataset = dataset.cache()
        if shuffle_on_memory > 0:
            dataset = dataset.shuffle(shuffle_on_memory, reshuffle_each_iteration=True)
        dataset = dataset.batch(config['batch_size'])
        dataset = dataset.prefetch(buffer_size=config['batch_size'])
        dataset = dataset.repeat(epoch_num)
        return dataset

    def collect_data(files, input_parser, split, take_these_splits):
        file_list = tf.data.Dataset.list_files(files, shuffle=True, seed=24)
        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(input_parser,)# num_parallel_calls=8)
        if split is None:
            return dataset
        else:
            datasets = split_dataset(dataset, split)
            dataset = datasets[take_these_splits[0]]
            if len(take_these_splits) > 2:
                for i in range(1, take_these_splits):
                    dataset = dataset.concatenate(datasets[take_these_splits[i]])
            return dataset

    dataset = collect_data(files, input_parser, split, take_these_splits)
    num_elements = 0
    for d in dataset:
        num_elements += 1
    if num_elements == 0:
        input_dim = None
    else:
        input_dim = d[0]['size'][1].numpy()
    info = {'num_elements': num_elements,
            'input_dim': input_dim}

    return input_fn, info


def train(config):
    with tf.io.gfile.GFile(
        tf.io.gfile.glob(os.path.join(os.path.dirname(config['dataset']), "tasks.txt"))[0], "r"
    ) as text_file:
        task_names = text_file.readlines()
    task_num = len(task_names)
    config['task_names'] = task_names
    config['task_num'] = task_num

    feature_spec = {
        "adj_column": tf.io.VarLenFeature(tf.int64),
        "adj_degrees": tf.io.VarLenFeature(tf.int64),
        "adj_elem_len": tf.io.FixedLenFeature([1], tf.int64),
        "adj_row": tf.io.VarLenFeature(tf.int64),
        "adj_values": tf.io.VarLenFeature(tf.float32),
        "feature_column": tf.io.VarLenFeature(tf.int64),
        "feature_elem_len": tf.io.FixedLenFeature([1], tf.int64),
        "feature_row": tf.io.VarLenFeature(tf.int64),
        "feature_values": tf.io.VarLenFeature(tf.float32),
        "label": tf.io.FixedLenFeature([task_num], tf.int64),
        "mask_label": tf.io.FixedLenFeature([task_num], tf.int64),
        "size": tf.io.FixedLenFeature([2], tf.int64),
    }
    input_parser = lambda example_proto: make_parse_fn(example_proto, feature_spec)
    shuffle_on_memory = 1000

    if config["mode"] == "train_cv":
        folds = config['k-fold_num']
        split = [1] * folds
        valid_dataset = config["dataset"]
    elif config["validation_dataset"] is None:
        folds = 1
        split = [100 - 100 * config['validation_data_rate'], 100 * config['validation_data_rate']]
        split = [int(s) for s in split]
        divisor = math.gcd(split[0], split[1])
        split = [s // divisor for s in split]
        train_portions = [0]
        valid_portions = [1]
        valid_dataset = config["dataset"]
    else:
        folds = 1
        split = None
        train_portions = None
        valid_portions = None
        valid_dataset = config["validation_dataset"]

    for fold_num in range(folds):
        if config["mode"] == "train_cv":
            train_portions = list(range(folds)) - fold_num
            valid_portions = [fold_num]
            config["model_dir"] = config["job_dir"] + "_fold_" + str(fold_num)
        else:
            config["model_dir"] = config["job_dir"]

        train_input_fn, info = make_input_fn(
            config["dataset"], input_parser, True, shuffle_on_memory, config['epoch'], split, train_portions)
        valid_input_fn, valid_info = make_input_fn(
            valid_dataset, input_parser, True, 0, 1, split, valid_portions)
        config['input_dim'] = info['input_dim']

        steps_per_epoch = math.ceil(info['num_elements'] / config['batch_size'])
        tf.logging.info(f"example num: {info['num_elements']}, steps per epoch: {steps_per_epoch}")
        steps_per_epoch_eval = math.ceil(valid_info['num_elements'] / config['batch_size'])
        tf.logging.info(f"example num: {valid_info['num_elements']}, steps per epoch: {steps_per_epoch_eval}")

        config['steps_per_epoch'] = steps_per_epoch
        model = importlib.import_module(config["model.py"]).build(config)

        tf.io.gfile.makedirs(model.eval_dir())
        feature_spec_predict = feature_spec.copy()
        feature_spec_predict.pop("label")
        feature_spec_predict.pop("mask_label")
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec_predict)
        exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_receiver_fn, exports_to_keep=1)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
        if valid_info['num_elements'] > 0:
            eval_spec = tf.estimator.EvalSpec(
                input_fn=valid_input_fn,
                steps=steps_per_epoch_eval,
                throttle_secs=0,
                exporters=exporter,
            )
            tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
        else:
            t = time.time()
            model.train(train_input_fn)
            elapsed = time.time() - t
            print("elapsed time: {}".format(elapsed))
            sys.exit(0)
        checkpoint_path = tf.io.gfile.glob(os.path.join(model.model_dir, "export/best_exporter/*/variables"))[0]
        checkpoint_path = checkpoint_path + "/variables"
        metafile = tf.io.gfile.glob(os.path.join(model.model_dir, "*.meta"))[-1]
        tf.io.gfile.copy(metafile, checkpoint_path + ".meta", overwrite=True)
        test_result = model.evaluate(
            input_fn=valid_input_fn,
            steps=steps_per_epoch_eval,
            checkpoint_path=checkpoint_path,
        )
        tf.io.gfile.mkdir(os.path.join(model.model_dir, "test"))
        test_result = {k: np.float(v) for k, v in test_result.items()}
        with tf.io.gfile.GFile(os.path.join(model.model_dir, "test", "test.json"), "w") as f:
            json.dump(test_result, f)


def _between(tensor, lower, upper):
    lower_bound = tf.math.greater_equal(tensor, lower)
    upper_bound = tf.math.less(tensor, upper)
    return tf.math.logical_and(lower_bound, upper_bound)


def split_dataset(dataset: tf.data.Dataset, split: Iterable[int], buffer_shuffle: int = None) -> List[tf.data.Dataset]:
    partitions = np.insert(np.cumsum(split), 0, 0)
    if buffer_shuffle is None:
        buffer_shuffle = 100 * partitions[-1]
    dataset = dataset.shuffle(buffer_shuffle, seed=22, reshuffle_each_iteration=False).enumerate()
    partitions = np.stack([partitions[:-1], partitions[1:]], axis=1)
    datasets = map(
        lambda partition: dataset.filter(
            lambda x, y: _between(
                tf.math.floormod(x, partitions[-1, -1]), partition[0], partition[1]
            )
        ).map(lambda x, y: y),
        partitions,
    )
    return list(datasets)


def infer(config):
    print("Under the construction")
    pass


if __name__ == "__main__":
    seed = 1234
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='train/infer/train_cv/visualize')
    parser.add_argument('--config', type=str, default=None, nargs='?',
                        help='config json file')
    parser.add_argument('--save-config', default=None, nargs='?',
                        help='save config json file')
    parser.add_argument('--retrain', type=str, default=None,
                        help='retrain from checkpoint')
    parser.add_argument('--no-config', action='store_true',
                        help='use default setting')
    parser.add_argument('--model', type=str, default=None,
                        help='model')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset')
    parser.add_argument('--gpu', type=str, default=None,
                        help='constraint gpus (default: all) (e.g. --gpu 0,2)')
    parser.add_argument('--cpu', action='store_true',
                        help='cpu mode (calcuration only with cpu)')
    parser.add_argument('--bspmm', action='store_true',
                        help='bspmm')
    parser.add_argument('--bconv', action='store_true',
                        help='bconv')
    parser.add_argument('--batched', action='store_true',
                        help='batched')
    parser.add_argument('--profile', action='store_true',
                        help='')
    parser.add_argument('--skfold', action='store_true',
                        help='stratified k-fold')
    parser.add_argument('--param', type=str, default=None,
                        help='parameter')
    parser.add_argument('--ig_targets', type=str, default='all',
                        choices=['all', 'profeat', 'features', 'adjs', 'dragon'],
                        help='[deplicated (use ig_modal_target)]set scaling targets for Integrated Gradients')
    parser.add_argument('--ig_modal_target', type=str, default='all',
                        choices=['all', 'profeat', 'features', 'adjs', 'dragon'],
                        help='set scaling targets for Integrated Gradients')
    parser.add_argument('--ig_label_target', type=str, default='max',
                        help='[visualization mode only] max/all/(label index)')
    parser.add_argument('--job_dir', type=str, default='train',
                        help='Directory in which log is stored.')
    args = parser.parse_args()

    config = get_default_config()
    if args.config is None:
        pass
    else:
        print("[LOAD] ", args.config)
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    if args.model is not None:
        config["load_model"] = args.model
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.param is not None:
        config["param"] = args.param
    if args.retrain is not None:
        config["retrain"] = args.retrain
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    elif args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #
    if args.profile:
        config["profile"] = True
    if args.skfold is not None:
        config["stratified_kfold"] = args.skfold
    if args.ig_targets != "all":
        args.ig_model_target = args.ig_targets
    config['job_dir'] = args.job_dir
    config["mode"] = args.mode
    if args.mode in ["train", "train_cv"]:
        train(config)
    elif args.mode == "infer":
        infer(config)
    if args.save_config is not None:
        print("[SAVE] ", args.save_config)
        os.makedirs(os.path.dirname(args.save_config), exist_ok=True)
        with open(args.save_config, "w") as f:
            json.dump(config, f, indent=4, cls=NumPyArangeEncoder)
