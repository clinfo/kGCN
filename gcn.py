import time
import json
import argparse
import importlib
import os
import sys

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, jaccard_score, \
    roc_curve, auc, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.python.framework import graph_util

import kgcn.layers
from kgcn.data_util import load_and_split_data, load_data, split_data
from kgcn.core import CoreModel
from kgcn.make_plots import plot_cost, plot_auc, plot_r2
from kgcn.make_plots import make_cost_acc_plot


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


def save_prediction(filename, prediction_data):
    print(f"[SAVE] {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pred = np.array(prediction_data)
    with open(filename, "w") as fp:
        if len(pred.shape) == 2:
            # graph-centric mode
            # prediction: graph_num x dist
            for dist in pred:
                fp.write(",".join(map(str, dist)))
                fp.write("\n")
        elif len(pred.shape) == 3:
            # node-centric mode
            # prediction: graph_num x node_num x dist
            for node_pred in pred:
                for dist in node_pred:
                    fp.write(",".join(map(str, dist)))
                    fp.write("\n")
                fp.write("\n")
        else:
            print("[ERROR] unknown prediction format")


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
    # config["save_result_train"]=None
    config["save_result_valid"] = None
    config["save_result_test"] = None
    config["save_result_cv"] = None
    config["save_info_train"] = None
    config["save_info_valid"] = None
    config["save_info_test"] = None
    config["save_info_cv"] = None
    config["make_plot"] = False
    config["plot_path"] = "./result/"
    config["visualize_path"] = "./visualization/"
    config["plot_multitask"] = False
    config["task"] = "multitask_classification"
    config["retrain"] = None
    #
    config["profile"] = False
    config["export_model"] = None
    # for visualization options
    config["visualize_kg"] = None

    config["stratified_kfold"] = False
    config["prediction_data"] = None

    return config


def load_model_py(model, model_py, is_train=True, feed_embedded_layer=False, batch_size=None):
    pair = model_py.split(":")
    sys.path.append(os.getcwd())
    if len(pair) >= 2:
        tf.logging.info(f"[LOAD] {pair[1]} from {pair[0]}")
        mod = importlib.import_module(pair[0])
        cls = getattr(mod, pair[1])
        obj = cls()
        if model:
            model.build(obj, is_train, feed_embedded_layer, batch_size)
        return obj
    else:
        tf.logging.info(f"[LOAD] {pair[0]}")
        mod = importlib.import_module(pair[0])
        if model:
            model.build(mod, is_train, feed_embedded_layer, batch_size)
        return mod


def print_ckpt(sess, ckpt):
    print(f"== {ckpt}")
    for var_name, _ in tf.contrib.framework.list_variables(ckpt):
        var = tf.contrib.framework.load_variable(ckpt, var_name)
        print(var_name, var.shape)
    print("==")


def print_variables():
    print('== neural network')
    vars_em = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in vars_em:
        print(v.name, v.shape)
    print("==")


def compute_metrics(config, info, prediction_data, labels):
    pred_score = np.array(prediction_data)
    true_label = np.array(labels)
    # pred_score: #data x # task x #class
    if len(pred_score.shape) == 1:
        pred_score = pred_score[:, np.newaxis, np.newaxis]
    elif len(pred_score.shape) == 2:
        pred_score = np.expand_dims(pred_score, axis=1)
    tf.logging.info(f"prediction #data x # task x #class: {pred_score.shape}")
    # multilabel=True  => pred_score: #data x # task x #class
    # multilabel=False => pred_score: #data x # task
    multiclass = False
    ntask = pred_score.shape[1]
    if pred_score.shape[2] == 1:  # regression or binary
        pred_score = pred_score[:, :, 0]
        tf.logging.info(f"2-class sigmoid")
    elif pred_score.shape[2] == 2:  # binary
        pred_score = pred_score[:, :, 1]
        tf.logging.info(f"2-class softmax")
    elif pred_score.shape[2] > 2:
        multiclass = True
        tf.logging.info(f"multi-class softmax")
    # true_label: #data x # task/#class
    if ntask == 1 and len(true_label.shape) == 2 and true_label.shape[1] == 2:
        true_label = true_label[:, 1]
    if len(true_label.shape) == 1:
        true_label = true_label[:, np.newaxis]

    tf.logging.info(f"label #data x # task/#class: {true_label.shape}")
    if not multiclass:
        tf.logging.info(f"binary-class mode")
        v = []
        for i in range(ntask):
            el = {}
            if config["task"] == "regression":
                el["r2"] = sklearn.metrics.r2_score(true_label[:, i], pred_score[:, i])
                el["mse"] = sklearn.metrics.mean_squared_error(true_label[:, i], pred_score[:, i])
            elif config["task"] == "regression_gmfe":
                el["gmfe"] = np.exp(np.mean(np.log(true_label[:, i]/pred_score[:, i])))
            else:
                pred = np.zeros(pred_score.shape)
                pred[pred_score > 0.5] = 1
                fpr, tpr, _ = roc_curve(true_label[:, i], pred_score[:, i], pos_label=1)
                roc_auc = auc(fpr, tpr)
                ap = average_precision_score(true_label[:, i], pred_score[:, i], pos_label=1)
                acc = accuracy_score(true_label[:, i], pred[:, i])
                scores = precision_recall_fscore_support(true_label[:, i], pred[:, i], average='binary')
                el["auc"] = roc_auc
                el["acc"] = acc
                el["ap"] = ap
                el["pre"] = scores[0]
                el["rec"] = scores[1]
                el["f"] = scores[2]
                el["sup"] = scores[3]
                el["balanced_acc"] = balanced_accuracy_score(true_label[:, i], pred[:, i])
                el["mcc"] = matthews_corrcoef(true_label[:, i], pred[:, i])
                try:
                    el["jaccard"] = jaccard_score(true_label[:, i], pred[:, i])
                except:
                    pass
            v.append(el)
    else:  # multiclass=True
        # #data x # task x #class
        # limitation: #task=1
        tf.logging.info(f"multi-class mode")
        pred = np.argmax(pred_score, axis=-1)
        true_label = np.argmax(true_label, axis=-1)
        pred = pred[:, 0]
        nclass = pred_score.shape[2]
        v = []
        for i in range(ntask):
            el = {}
            acc = accuracy_score(true_label, pred)
            scores = precision_recall_fscore_support(true_label, pred, labels=list(range(nclass)), average=None)
            el["acc"] = acc
            el["pre"] = scores[0]
            el["rec"] = scores[1]
            el["f"] = scores[2]
            el["sup"] = scores[3]
            el["balanced_acc"] = balanced_accuracy_score(true_label, pred)
            el["mcc"] = matthews_corrcoef(true_label, pred)
            try:
                el["jaccard"] = jaccard_score(true_label, pred)
            except:
                pass
            v.append(el)
    return v


def train(sess, graph, config):
    if config["validation_dataset"] is None:
        _, train_data, valid_data, info = load_and_split_data(config, filename=config["dataset"],
                                                              valid_data_rate=config["validation_data_rate"])
    else:
        print("[INFO] training")
        train_data, info = load_data(config, filename=config["dataset"])
        print("[INFO] validation")
        valid_data, valid_info = load_data(config, filename=config["validation_dataset"])
        info["graph_node_num"] = max(info["graph_node_num"], valid_info["graph_node_num"])
        info["graph_num"] = info["graph_num"] + valid_info["graph_num"]

    model = CoreModel(sess, config, info)
    load_model_py(model, config["model.py"])

    metric_name = ("mse" if config["task"] == "regression" else
                   "gmfe" if config["task"] == "regression_gmfe" else
                   "accuracy")

    if config["profile"]:
        vars_to_train = tf.trainable_variables()
        print(vars_to_train)

    # Training
    start_t = time.time()
    model.fit(train_data, valid_data)
    train_time = time.time() - start_t
    print(f"training time: {train_time}[sec]")
    if valid_data.num > 0:
        # Validation
        start_t = time.time()
        valid_cost, valid_metrics, prediction_data = model.pred_and_eval(valid_data)
        infer_time = time.time() - start_t
        print(f"final cost = {valid_cost}\n"
              f"{metric_name} = {valid_metrics[metric_name]}\n"
              f"validation time: {infer_time}[sec]\n")
        # Saving
        if config["save_info_valid"] is not None:
            result = {}
            result["validation_cost"] = valid_cost
            result["validation_accuracy"] = valid_metrics
            result["train_time"] = train_time
            result["infer_time"] = infer_time
            result["valid_metrics"] = compute_metrics(config, info, prediction_data, valid_data.labels)
            ##
            save_path = config["save_info_valid"]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"[SAVE] {save_path}")
            with open(save_path, "w") as fp:
                json.dump(result, fp, indent=4, cls=NumPyArangeEncoder)

    if config["export_model"]:
        try:
            print(f"[SAVE] {config['export_model']}")
            graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
            tf.train.write_graph(graph_def, '.', config["export_model"], as_text=False)
        except:
            print('[ERROR] output has been not found')
    if config["save_result_valid"] is not None:
        filename = config["save_result_valid"]
        save_prediction(filename, prediction_data)
    if config["make_plot"]:
        if config["task"] == "regression" or config["task"] == "regression_gmfe":
            # plot_cost(config, valid_data, model)
            plot_r2(config, valid_data.labels, np.array(prediction_data))
        else:
            plot_cost(config, valid_data, model)
            plot_auc(config, valid_data.labels, np.array(prediction_data))


def train_cv(sess, graph, config):
    all_data, info = load_data(config, filename=config["dataset"], prohibit_shuffle=True)  # shuffle is done by KFold
    model = CoreModel(sess, config, info)
    load_model_py(model, config["model.py"])
    # Training
    if config["stratified_kfold"]:
        print("[INFO] use stratified K-fold")
        kf = StratifiedKFold(n_splits=config["k-fold_num"], shuffle=config["shuffle_data"], random_state=123)
    else:
        kf = KFold(n_splits=config["k-fold_num"], shuffle=config["shuffle_data"], random_state=123)

    kf_count = 1
    fold_data_list = []
    output_data_list = []
    if all_data["labels"] is not None:
        split_base = all_data["labels"]
    else:
        split_base = all_data["label_list"][0]
    if config["stratified_kfold"]:
        split_base = np.argmax(split_base, axis=1)
    score_metrics = []
    if config["task"] == "regression":
        metric_name = "mse"
    elif config["task"] == "regression_gmfe":
        metric_name = "gmfe"
    else:
        metric_name = "accuracy"
    split_data_generator = kf.split(split_base, split_base) if config["stratified_kfold"] else kf.split(split_base)
    for train_valid_list, test_list in split_data_generator:
        print(f"starting fold: {kf_count}")
        train_valid_data, test_data = split_data(all_data,
                                                 indices_for_train_data=train_valid_list,
                                                 indices_for_valid_data=test_list)

        train_data, valid_data = split_data(train_valid_data, valid_data_rate=config["validation_data_rate"])
        # Training
        print(train_valid_list)
        print(test_list)
        start_t = time.time()
        model.fit(train_data, valid_data, k_fold_num=kf_count)
        train_time = time.time() - start_t
        print(f"training time: {train_time}[sec]")
        # Test
        print("== valid data ==")
        start_t = time.time()
        valid_cost, valid_metrics, prediction_data = model.pred_and_eval(valid_data)
        infer_time = time.time() - start_t
        print(f"final cost = {valid_cost}\n"
              f"{metric_name} = {valid_metrics[metric_name]}\n"
              f"infer time: {infer_time}[sec]\n")
        print("== test data ==")
        start_t = time.time()
        test_cost, test_metrics, prediction_data = model.pred_and_eval(test_data)
        infer_time = time.time() - start_t
        print(f"final cost = {test_cost}\n"
              f"{metric_name} = {test_metrics[metric_name]}\n")
        score_metrics.append(test_metrics[metric_name])
        print(f"infer time: {infer_time}[sec]")

        if config["export_model"]:
            try:
                name, ext = os.path.splitext(config["export_model"])
                filename = name+"."+str(kf_count)+ext
                print(f"[SAVE] {filename}")
                graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
                tf.train.write_graph(graph_def, '.', filename, as_text=False)
            except:
                print('[ERROR] output has been not found')
        if "save_edge_result_cv" in config:
            output_data = model.output(test_data)
            output_data_list.append(output_data)
        # save fold data
        fold_data = dotdict({})
        fold_data.prediction_data = prediction_data
        if all_data["labels"] is not None:
            fold_data.test_labels = test_data.labels
        else:
            fold_data.test_labels = test_data.label_list
        fold_data.test_data_idx = test_list
        if config["task"] == "regression":
            fold_data.training_mse = [el["training_mse"] for el in model.training_metrics_list]
            fold_data.validation_mse = [el["validation_mse"] for el in model.validation_metrics_list]
        elif config["task"] == "regression_gmfe":
            fold_data.training_mse = [el["training_gmfe"] for el in model.training_metrics_list]
            fold_data.validation_mse = [el["validation_gmfe"] for el in model.validation_metrics_list]
        else:
            fold_data.training_acc = [el["training_accuracy"] for el in model.training_metrics_list]
            fold_data.validation_acc = [el["validation_accuracy"] for el in model.validation_metrics_list]
        fold_data.test_acc = test_metrics[metric_name]
        fold_data.training_cost = model.training_cost_list
        fold_data.validation_cost = model.validation_cost_list
        fold_data.test_cost = test_cost
        fold_data.train_time = train_time
        fold_data.infer_time = infer_time
        fold_data_list.append(fold_data)
        kf_count += 1

    print(f"cv {metric_name}(mean) = {np.mean(score_metrics)}\n"
          f"cv {metric_name}(std.)   = {np.std(score_metrics)}\n")
    if "save_info_cv" in config and config["save_info_cv"] is not None:
        save_path = config["save_info_cv"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[SAVE] {save_path}")
        _, ext = os.path.splitext(save_path)
        if ext == ".json":
            with open(save_path, "w") as fp:
                json.dump(fold_data_list, fp, indent=4, cls=NumPyArangeEncoder)
        else:
            joblib.dump(fold_data_list, save_path, compress=True)
    #
    if "save_edge_result_cv" in config and config["save_edge_result_cv"] is not None:
        result_cv = []
        for j, fold_data in enumerate(fold_data_list):
            pred_score = np.array(fold_data.prediction_data)
            true_label = np.array(fold_data.test_labels)
            test_idx = fold_data.test_data_idx
            score_list = []
            for pair in true_label[0]:
                i1, _, j1, i2, _, j2 = pair
                s1 = pred_score[0, i1, j1]
                s2 = pred_score[0, i2, j2]
                score_list.append([s1, s2])
            fold = {}
            fold["output"] = output_data_list[j][0]
            fold["score"] = np.array(score_list)
            fold["test_data_idx"] = test_idx
            result_cv.append(fold)
        save_path = config["save_edge_result_cv"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[SAVE] {save_path}")
        _, ext = os.path.splitext(save_path)
        if ext == ".json":
            with open(save_path, "w") as fp:
                json.dump(result_cv, fp, indent=4, cls=NumPyArangeEncoder)
        else:
            joblib.dump(result_cv, save_path, compress=True)
    #
    if "save_result_cv" in config and config["save_result_cv"] is not None:
        result_cv = []
        for j, fold_data in enumerate(fold_data_list):
            v = compute_metrics(config, info, fold_data.prediction_data, fold_data.test_labels)
            result_cv.append(v)
        save_path = config["save_result_cv"]
        print(f"[SAVE] {save_path}")
        with open(save_path, "w") as fp:
            json.dump(result_cv, fp, indent=4, cls=NumPyArangeEncoder)
    #
    for i, fold_data in enumerate(fold_data_list):
        prefix = "fold"+str(i)+"_"
        result_path = config["plot_path"]
        os.makedirs(result_path, exist_ok=True)
        if config["make_plot"]:
            if config["task"] == "regression":
                make_cost_acc_plot(fold_data.training_cost, fold_data.validation_cost,
                                   fold_data.training_mse, fold_data.validation_mse, result_path+prefix)
                pred_score = np.array(fold_data.prediction_data)
                plot_r2(config, fold_data.test_labels, pred_score, prefix=prefix)
            elif config["task"] == "regression_gmfe":
                make_cost_acc_plot(fold_data.training_cost, fold_data.validation_cost,
                                   fold_data.training_mse, fold_data.validation_mse, result_path+prefix)
                pred_score = np.array(fold_data.prediction_data)
                plot_r2(config, fold_data.test_labels, pred_score, prefix=prefix)
            else:
                make_cost_acc_plot(fold_data.training_cost, fold_data.validation_cost,
                                   fold_data.training_acc, fold_data.validation_acc, result_path+prefix)
                pred_score = np.array(fold_data.prediction_data)
                plot_auc(config, fold_data.test_labels, pred_score, prefix=prefix)


def infer(sess, graph, config):
    dataset_filename = config["dataset"]
    if "dataset_test" in config:
        dataset_filename = config["dataset_test"]
    all_data, info = load_data(config, filename=dataset_filename, prohibit_shuffle=True)

    model = CoreModel(sess, config, info)
    load_model_py(model, config["model.py"], is_train=False)

    metric_name = ("mse" if config["task"] == "regression" else
                   "gmfe" if config["task"] == "regression_gmfe" else
                   "accuracy")

    # Initialize session
    restore_ckpt(sess, config["load_model"])

    # Validation
    start_t = time.time()
    test_cost, test_metrics, prediction_data = model.pred_and_eval(all_data)
    infer_time = time.time() - start_t
    print(f"final cost = {test_cost}\n"
          f"{metric_name} = {test_metrics[metric_name]}\n"
          f"infer time: {infer_time}[sec]\n")

    if config["save_info_test"] is not None:
        result = {}
        result["test_cost"] = test_cost
        result["test_accuracy"] = test_metrics
        result["infer_time"] = infer_time
        result["test_metrics"] = compute_metrics(config, info, prediction_data, all_data.labels)
        save_path = config["save_info_test"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[SAVE] {save_path}")
        with open(save_path, "w") as fp:
            json.dump(result, fp, indent=4, cls=NumPyArangeEncoder)

    if config["save_result_test"] is not None:
        filename = config["save_result_test"]
        save_prediction(filename, prediction_data)
    if config["make_plot"]:
        plot_auc(config, all_data.labels, np.array(prediction_data))
    if "save_edge_result_test" in config and config["save_edge_result_test"] is not None:
        output_data = model.output(all_data)
        pred_score = np.array(prediction_data)
        true_label = np.array(all_data.label_list)
        score_list = []
        print(true_label.shape)
        for pair in true_label[0]:
            i1, _, j1, i2, _, j2 = pair
            s1 = pred_score[0, i1, j1]
            s2 = pred_score[0, i2, j2]
            score_list.append([s1, s2])
        fold = {}
        fold["output"] = output_data[0]
        fold["score"] = np.array(score_list)
        save_path = config["save_edge_result_test"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[SAVE] {save_path}")
        _, ext = os.path.splitext(save_path)
        if ext == ".json":
            with open(save_path, "w") as fp:
                json.dump(fold, fp, indent=4, cls=NumPyArangeEncoder)
        else:
            joblib.dump(fold, save_path, compress=True)
    if config["prediction_data"] is not None:
        obj = {}
        obj["prediction_data"] = prediction_data
        obj["labels"] = all_data.labels

        os.makedirs(os.path.dirname(config["prediction_data"]), exist_ok=True)
        joblib.dump(obj, config["prediction_data"])


def restore_ckpt(sess, ckpt):
    saver = tf.train.Saver()
    tf.logging.info(f"[LOAD]{ckpt}")
    try:
        saver.restore(sess, ckpt)
    except:
        print("======LOAD ERROR======")
        print_variables()
        print_ckpt(sess, ckpt)
        raise Exception
    return saver


def visualize(sess, config, args):
    from kgcn.visualization import cal_feature_IG, cal_feature_IG_for_kg
    # input a molecule at a time
    batch_size = 1
    dataset_filename = config["dataset"]
    if "dataset_test" in config:
        dataset_filename = config["dataset_test"]
    all_data, info = load_data(config, filename=dataset_filename, prohibit_shuffle=True)

    model = CoreModel(sess, config, info)
    load_model_py(model, config["model.py"], is_train=False, feed_embedded_layer=True, batch_size=batch_size)
    placeholders = model.placeholders
    restore_ckpt(sess, config['load_model'])
    # calculate integrated gradients
    if config['visualize_type'] == 'graph':
        cal_feature_IG(sess, all_data, placeholders, info, config, model.prediction,
                       args.ig_modal_target, args.ig_label_target,
                       logger=tf.logging, model=model.nn, args=args)
    else:
        cal_feature_IG_for_kg(sess, all_data, placeholders, info, config, model.prediction,
                              logger=tf.logging, model=model.nn, args=args)


def main():
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
                        choices=['all', 'profeat', 'features', 'adjs', 'dragon','embedded_layer'],
                        help='[deplicated (use ig_modal_target)]set scaling targets for Integrated Gradients')
    parser.add_argument('--ig_modal_target', type=str, default='all',
                        choices=['all', 'profeat', 'features', 'adjs', 'dragon','embedded_layer'],
                        help='set scaling targets for Integrated Gradients')
    parser.add_argument('--ig_label_target', type=str, default='max',
                        help='[visualization mode only] max/all/(label index)')
    parser.add_argument('--visualize_type', type=str, default='graph',
                        choices=['graph', 'node', 'edge_loss', 'edge_score'],
                        help="graph: visualize graph's property. node: create an integrated gradients map"
                             " using target node. edge_loss: create an integrated gradients map"
                             " using target edge and loss function. edge_score: create an integrated gradients map"
                             " using target edge and score function.")
    parser.add_argument('--visualize_target', type=int, default=None,
                        help="set the target's number you want to visualize. from: [0, ~)")
    parser.add_argument('--visualize_resample_num', type=int, default=None,
                        help="resampling for visualization: [0, ~v)")
    parser.add_argument('--visualize_method', type=str, default='ig',
                        choices=['ig', 'grad', 'grad_prod', 'smooth_grad', 'smooth_ig'],
                        help="visualization methods")
    parser.add_argument('--graph_distance', type=int, default=1,
                        help=("set the distance from target node. An output graph is created within "
                              "the distance from target node. :[1, ~)"))
    parser.add_argument('--verbose', action="store_true",
                        help="set log level")
    parser.add_argument('--visualization_header', type=str, default=None,
                        help="filename header of visualization")

    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.WARN)

    # config
    config = get_default_config()
    if args.config is None:
        pass
    else:
        print(f"[LOAD] {args.config}")
        with open(args.config, "r") as fp:
            config.update(json.load(fp))
    # option
    if args.model is not None:
        config["load_model"] = args.model
    if args.dataset is not None:
        config["dataset"] = args.dataset
    # param
    if args.param is not None:
        config["param"] = args.param
    # option
    if args.retrain is not None:
        config["retrain"] = args.retrain
    # gpu/cpu
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    elif args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #
    if args.profile:
        config["profile"] = True
    if args.skfold is not None:
        config["stratified_kfold"] = args.skfold
    # bspmm
    # if args.disable_bspmm:
    #    print("[INFO] disabled bspmm")
    # else:
    kgcn.layers.load_bspmm(args)
    # print("[INFO] enabled bspmm")
    # depricated options
    if args.ig_targets != "all":
        args.ig_modal_target = args.ig_targets
    # setup

    config["visualize_type"] = args.visualize_type
    config["visualize_target"] = args.visualize_target
    config["graph_distance"] = args.graph_distance

    with tf.Graph().as_default() as graph:
        seed = 1234
        tf.set_random_seed(seed)
        with tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                              gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # mode
            config["mode"] = args.mode
            if args.mode == "train":
                train(sess, graph, config)
            if args.mode == "train_cv":
                train_cv(sess, graph, config)
            elif args.mode == "infer" or args.mode == "predict":
                infer(sess, graph, config)
            elif args.mode == "visualize":
                visualize(sess, config, args)
        import tfcg
        parser = tfcg.from_graph_def(sess.graph_def)
        logdir = 'logdir'
        writer = tf.summary.FileWriter(logdir, sess.graph)
        parser.dump_img("output.png")
        parser.dump_yml("output.yml")
    if args.save_config is not None:
        print(f"[SAVE] {args.save_config}")
        os.makedirs(os.path.dirname(args.save_config), exist_ok=True)
        with open(args.save_config, "w") as fp:
            json.dump(config, fp, indent=4, cls=NumPyArangeEncoder)


if __name__ == '__main__':
    main()
