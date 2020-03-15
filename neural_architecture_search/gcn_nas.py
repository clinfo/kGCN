#!/usr/bin/env python
import sys
import uuid
import typing
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow.examples.tutorials.mnist import input_data


from tfdbonas import Searcher, Trial
from tnng import Generator, MultiHeadLinkedListLayer
import tfcg

# kgcn library
sys.path.append(Path('..').resolve())
from gcn import get_default_config
from kgcn.data_util import load_and_split_data
from kgcn.core import CoreModel
from kgcn.default_model import DefaultModel
import kgcn.layers



def _get_config():
    config = get_default_config()
    params = {
        "model.py": "example_model.model_multimodal:GCN",
        "save_result_test": "../result/test.multimodal.csv",
        "save_result_train": "../result/train.multimodal.csv",
        "load_model": "../model/model.sample_multimodal.ckpt",
        "save_model": "../model/model.sample_multimodal.ckpt",
        "validation_data_rate": 0.3,
        "embedding_dim":4,
        "epoch": 1,
        "with_feature": True,
        "batch_size": 10,
        "save_interval": 10,
        "learning_rate": 0.3,
        "with_node_embedding": False,
        "save_model_path": "model",
        "patience": 0,
        "dataset": "../example_jbl/sample.jbl"
    }
    for k, v in params.items():
        config[k] = v
    return config



class GCN(DefaultModel):
    def __init__(self, layers):
        self.layers = layers

    def build_placeholders(self, info, config, batch_size, **kwargs):
        # input data types (placeholders) of this neural network
        keys = ['adjs', 'nodes', 'labels', 'mask', 'dropout_rate', 'enabled_node_nums', 'is_train', 'features',
                'sequences', 'sequences_len', 'embedded_layer']
        return self.get_placeholders(info, config, batch_size, keys, **kwargs)

    def build_model(self, placeholders, info, config, batch_size, feed_embedded_layer=False, **kwargs):
        adj_channel_num = info.adj_channel_num
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        in_nodes = placeholders["nodes"]
        labels = placeholders["labels"]
        mask = placeholders["mask"]
        enabled_node_nums = placeholders["enabled_node_nums"]
        is_train = placeholders["is_train"]
        dropout_rate = placeholders["dropout_rate"]
        sequences = placeholders["sequences"]
        sequences_len = placeholders["sequences_len"]
        embedded_layer = placeholders["embedded_layer"]

        ###
        ### Graph part
        ###
        #with tf.variable_scope("graph_nn") as scope_part:
        layer = features
        print('layer::::::::', layer)
        input_dim = info.feature_dim
        # layer: batch_size x graph_node_num x dim

        xx = [layer, sequences]
        for _layer in self.layers:
            if len(_layer) == 2:
                if _layer[0] is None:
                    x1 = xx[0]
                    x2 = _layer[1](xx[1])
                elif _layer[1] is None:
                    x1 = _layer[0](xx[0])
                    x1 = tf.math.sigmoid(x1)
                    x2 = xx[1]
                else:
                    if isinstance(_layer[0], kgcn.layers.GraphConv):
                        x1 = _layer[0](xx[0], adj=in_adjs)
                    else:
                        x1 = _layer[0](xx[0])
                        x1 = tf.math.sigmoid(x1)
                    x2 = _layer[1](xx[1])
                xx = [x1, x2]
            elif len(_layer) == 1:
                if _layer[0] == 'concat':
                    xx = keras.layers.concatenate(xx, axis=1)
                else:
                    xx = _layer[0](xx)
        layer = xx
        prediction = layer
        # computing cost and metrics
        cost = mask*tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=layer)
        cost_opt = tf.reduce_mean(cost)

        metrics = {}
        cost_sum = tf.reduce_sum(cost)

        correct_count = mask*tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32)
        metrics["correct_count"] = tf.reduce_sum(correct_count)
        self.out = layer
        return self, prediction, cost_opt, cost_sum, metrics

    def embedding(self, sess, data):
        key = self.placeholders['sequences']
        feed_dict = {key: data}
        out = sess.run(self.embedding_layer, feed_dict)
        return out

def _train(layers):
    config = _get_config()
    _, train_data, valid_data, info = load_and_split_data(config, filename=config["dataset"],
                                                          valid_data_rate=config["validation_data_rate"])
    metric_name = "accuracy"
    with tf.Session() as sess:
        model = CoreModel(sess, config, info)
        model.build(GCN(layers), True, False, None)
        model.fit(train_data, valid_data)
        _, valid_metrics, _ = model.pred_and_eval(valid_data)
    return valid_metrics[metric_name]

def objectve(trial: Trial):
    layers, (_, _) = trial.graph
    accuracy = _train(layers)
    return accuracy

def gcn_model():
    m = MultiHeadLinkedListLayer()
    # graph created
    gcn_args = [dict(output_dim=i, adj_channel_num=1) for i in [32, 48, 64]]
    gcn_dense_args = [dict(output_dim=i) for i in [32, 48, 64, 128]]
    m.append_lazy(kgcn.layers.GraphConv, gcn_args)
    m.append_lazy(kgcn.layers.GraphDense, gcn_dense_args)
    m.append_lazy(kgcn.layers.GraphGather, [dict(),])
    return m

def linear_model(input_dim):
    m = MultiHeadLinkedListLayer()
    emb_args = [dict(input_dim=input_dim, output_dim=i) for i in [50,]]
    m.append_lazy(keras.layers.Embedding, emb_args)
    conv_args = [dict(filters=i, kernel_size=4, padding="same", activation='relu') for i in [32, 48, 64]]
    m.append_lazy(keras.layers.Conv1D, conv_args)
    m.append_lazy(keras.layers.MaxPooling1D, [dict(pool_size=4),])
    lstm_args = [dict(units=i, return_sequences=False, go_backwards=True) for i in [32, 48, 64]]
    m.append_lazy(keras.layers.LSTM, lstm_args)
    return m


if __name__ == '__main__':
    config = _get_config()
    _, train_data, valid_data, info = load_and_split_data(config, filename=config["dataset"],
                                                          valid_data_rate=config["validation_data_rate"])
    m1 = gcn_model()
    m2 = linear_model(info.sequence_symbol_num)
    m = m1 + m2
    dense_args = [dict(units=i, activation='relu') for i in [32, 64, 128]]
    m.append_lazy(keras.layers.Dense, dense_args)
    m.append_lazy(keras.layers.Dense, [dict(units=info.label_dim, activation='softmax'),])
    g = Generator(m, dump_nn_graph=True)
    num_nodes = 12
    num_layer_type = 4
    searcher = Searcher()
    searcher.register_trial('graph', g)
    n_trials = 30
    model_kwargs = dict(
        num_nodes=num_nodes,
        input_channels=num_layer_type,
        n_train_epochs=400,
    )
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model=f'tfdbonas.deep_surrogate_models:GCNSurrogateModel',
                        n_random_trials=10,
                        model_kwargs=model_kwargs)
    print(searcher.result)
    print('best_trial', searcher.best_trial)
    print('best_value', searcher.best_value)
