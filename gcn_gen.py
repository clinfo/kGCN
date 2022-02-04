import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.logging as logging
import numpy as np
import joblib
import time
import json
import argparse
import os

from kgcn.gcn import NumPyArangeEncoder
from kgcn.gcn import get_default_config, load_model_py
from kgcn.data_util import load_and_split_data, load_data
from kgcn.core import CoreModel
from kgcn.feed_index import construct_feed


def print_ckpt(sess, ckpt):
    #checkpoint = tf.train.get_checkpoint_state(args.ckpt)
    print("==", ckpt)
    for var_name, _ in tf.contrib.framework.list_variables(ckpt):
        var = tf.contrib.framework.load_variable(ckpt, var_name)
        print(var_name, var.shape)
    print("==")


def print_variables():
    # print variables
    print('== neural network')
    vars_em = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in vars_em:
        print(v.name, v.shape)
    print("==")


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


def get_pos_weight(data):
    adjs = data.adjs
    ws = []
    for adj in adjs:
        for ch, a in enumerate(adj):
            num = a[2][0]
            num_all = num*num
            num_pos = len(a[0])
            num_neg = num_all-num_pos
            ws.append(num_neg/num_pos)
    return np.mean(ws)


def get_norm(data):
    adjs = data.adjs
    ws = []
    for adj in adjs:
        for ch, a in enumerate(adj):
            num = a[2][0]
            num_all = num*num
            num_pos = len(a[0])
            num_neg = num_all-num_pos
            ws.append(num_all/num_neg*2)
    return np.mean(ws)


def train(sess, config):
    if config["validation_dataset"] is None:
        all_data, train_data, valid_data, info = load_and_split_data(config, filename=config["dataset"],
                                                                     valid_data_rate=config["validation_data_rate"])
    else:
        print("[INFO] training")
        train_data, info = load_data(config, filename=config["dataset"])
        print("[INFO] validation")
        valid_data, valid_info = load_data(config, filename=config["validation_dataset"])
        info["graph_node_num"] = max(info["graph_node_num"], valid_info["graph_node_num"])
        info["graph_num"] = info["graph_num"] + valid_info["graph_num"]
    # train model
    graph_index_list = []
    for i in range(info["graph_num"]):
        graph_index_list.append([i, i])
    info.graph_index_list = graph_index_list
    info.pos_weight = get_pos_weight(train_data)
    info.norm = get_norm(train_data)
    print(f"pos_weight={info.pos_weight}")
    print(f"norm={info.norm}")

    model = CoreModel(sess, config, info, construct_feed_callback=construct_feed)
    load_model_py(model, config["model.py"])

    vars_to_train = tf.trainable_variables()
    for v in vars_to_train:
        print(v)

    # Training
    start_t = time.time()
    model.fit(train_data, valid_data)
    train_time = time.time() - start_t
    print(f"training time:{train_time}[sec]")
    # Validation
    start_t = time.time()
    validation_cost, validation_accuracy, validation_prediction_data = model.pred_and_eval(valid_data)
    training_cost, training_accuracy, training_prediction_data = model.pred_and_eval(train_data)
    infer_time = time.time() - start_t
    print(f"final cost(training  ) = {training_cost}\n"
          f"accuracy  (training  ) = {training_accuracy['accuracy']}\n"
          f"final cost(validation) = {validation_cost}\n"
          f"accuracy  (validation) = {validation_accuracy['accuracy']}\n"
          f"infer time:{infer_time}[sec]\n")
    # Saving
    if config["save_info_valid"] is not None:
        result = {}
        result["validation_cost"] = validation_cost
        result["validation_accuracy"] = validation_accuracy["accuracy"]
        result["train_time"] = train_time
        result["infer_time"] = infer_time
        save_path = config["save_info_valid"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[SAVE] {save_path}")
        with open(save_path, "w") as fp:
            json.dump(result, fp, indent=4)

    if config["save_info_train"] is not None:
        result = {}
        result["test_cost"] = training_cost
        result["test_accuracy"] = training_accuracy["accuracy"]
        result["train_time"] = train_time
        save_path = config["save_info_train"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[SAVE] {save_path}")
        with open(save_path, "w") as fp:
            json.dump(result, fp, indent=4, cls=NumPyArangeEncoder)

    if "reconstruction_valid" in config:
        filename = config["reconstruction_valid"]
        print(os.path.dirname(filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"[SAVE] {filename}")
        joblib.dump(validation_prediction_data, filename)
    if "reconstruction_train" in config:
        filename = config["reconstruction_train"]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"[SAVE] {filename}")
        joblib.dump(training_prediction_data, filename)


def reconstruct(sess, config):
    dataset_filename = config["dataset"]
    if "dataset_test" in config:
        dataset_filename = config["dataset_test"]
    all_data, info = load_data(config, filename=dataset_filename)

    graph_index_list = []
    for i in range(all_data.num):
        graph_index_list.append([i, i])
    info.graph_index_list = graph_index_list
    info.pos_weight = get_pos_weight(all_data)
    info.norm = get_norm(all_data)
    print(f"pos_weight={info.pos_weight}")
    print(f"norm={info.norm}")

    model = CoreModel(sess, config, info, construct_feed_callback=construct_feed)
    load_model_py(model, config["model.py"], is_train=False)

    vars_to_train = tf.trainable_variables()
    for v in vars_to_train:
        print(v)

    # initialize session
    restore_ckpt(sess, config["load_model"])

    start_t = time.time()
    cost, acc, pred_data = model.pred_and_eval(all_data)
    recons_data = pred_data
    """
    recons_data=[]
    for i in range(3):
        print(i)
        cost,acc,pred_data=model.pred_and_eval(all_data)
        recons_data.append(pred_data)
    """
    if "reconstruction_test" in config:
        filename = config["reconstruction_test"]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"[SAVE] {filename}")
        joblib.dump(recons_data, filename)


def generate(sess, config):
    dataset_filename = config["dataset"]
    if "dataset_test" in config:
        dataset_filename = config["dataset_test"]
    all_data, info = load_data(config, filename=dataset_filename)

    graph_index_list = []
    for i in range(all_data.num):
        graph_index_list.append([i, i])
    info.graph_index_list = graph_index_list
    info.pos_weight = get_pos_weight(all_data)
    info.norm = get_norm(all_data)
    print(f"pos_weight={info.pos_weight}")
    print(f"norm={info.norm}")

    model = CoreModel(sess, config, info, construct_feed_callback=construct_feed)
    load_model_py(model, config["model.py"], is_train=False)
    # initialize session
    saver = tf.train.Saver()
    #sess.run(tf.global_variables_initializer())
    restore_ckpt(sess, config["load_model"])

    start_t = time.time()
    cost, acc, pred_data = model.pred_and_eval(all_data)
    generated_data = pred_data

    if "generation_test" in config:
        filename = config["generation_test"]
        dirname = os.path.dirname(filename)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        print(f"[SAVE] {filename}")
        joblib.dump(generated_data, filename)


def main():
    seed = 1234
    np.random.seed(seed)
    tf.set_random_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='train/infer')
    parser.add_argument('--config', type=str,
                        default=None,
                        nargs='?',
                        help='config json file')
    parser.add_argument('--save-config',
                        default=None,
                        nargs='?',
                        help='save config json file')
    parser.add_argument('--no-config',
                        action='store_true',
                        help='use default setting')
    parser.add_argument('--model', type=str,
                        default=None,
                        help='model')
    parser.add_argument('--dataset', type=str,
                        default=None,
                        help='dataset')
    parser.add_argument('--gpu', type=str,
                        default=None,
                        help='constraint gpus (default: all) (e.g. --gpu 0,2)')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='cpu mode (calcuration only with cpu)')

    args = parser.parse_args()
    # config
    config = get_default_config()
    if args.config is None:
        pass
        #parser.print_help()
        #quit()
    else:
        print("[LOAD] ", args.config)
        fp = open(args.config, 'r')
        config.update(json.load(fp))
    # option
    if args.model is not None:
        config["load_model"] = args.model
    if args.dataset is not None:
        config["dataset"] = args.dataset
    # gpu/cpu
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    elif args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # setup
    with tf.Graph().as_default():
    #with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # mode
            if args.mode == "train":
                train(sess, config)
            elif args.mode == "reconstruct":
                reconstruct(sess, config)
            elif args.mode == "generate":
                generate(sess, config)
    if args.save_config is not None:
        print(f"[SAVE] {args.save_config}")
        fp = open(args.save_config, "w")
        json.dump(config, fp, indent=4)


if __name__ == '__main__':
    main()
