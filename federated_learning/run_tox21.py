#!/usr/bin/env python
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import tensorflow_federated as tff

import kgcn.layers as layers
from kgcn.data_util import load_and_split_data


def client_data(source, n):
    return source.create_tf_dataset_for_client(source.client_ids[n]).repeat(10).batch(20)

def build_model():
    input_features = tf.keras.Input(shape=(132, 81), name='features')
    input_adjs = tf.keras.Input(shape=(1, 132, 132), name='adjs', sparse=False)
    input_mask_label = tf.keras.Input(shape=(12), name="mask_label")
    h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(2, tf.nn.softmax, input_shape=[64])(h)

    return keras.Model(inputs=[input_features, input_adjs, input_mask_label], outputs=logits)
    

if __name__ == '__main__':
    N_CLIENTS = 2
    config = {"normalize_adj_flag": True, "with_feature": True, "split_adj_flag": False, "shuffle_data": False, "dataset": "dataset.jbl", "validation_data_rate": 0.2}
    _, train_data, valid_data, info = load_and_split_data(config, filename=config["dataset"],
                                                              valid_data_rate=config["validation_data_rate"])
    adjs = tf.sparse.concat(0, [tf.sparse.SparseTensor(train_data['adjs'][i][0][0], train_data['adjs'][i][0][1], train_data['adjs'][i][0][2]) for i in range(train_data.num)])
    adjs = tf.sparse.reshape(adjs, [train_data.num, 1, -1, adjs.shape[-1]])
    adjs = tf.sparse.to_dense(adjs)
    train_dataset = tff.simulation.FromTensorSlicesClientData({'bob': (OrderedDict({"features": train_data['features'], "adjs": adjs, "mask_label": train_data["mask_label"]}), np.ones(len(adjs)))})#train_data.labels[:, 0])})

    def client_data(n):
        return train_dataset.create_tf_dataset_for_client(n).repeat(10).batch(20)

    train_data = [client_data(n) for n in ['bob']]
    def model_fn():
        model = build_model()
        return tff.learning.from_keras_model(
            model,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            input_spec=train_data[0].element_spec)
            #metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    state = trainer.initialize()
    for _ in ['bob']:
        state, metrics = trainer.next(state, train_data)
        print(metrics)
        #print (metrics.loss)
