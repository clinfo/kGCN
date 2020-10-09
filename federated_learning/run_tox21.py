#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import tensorflow_federated as tff

import kgcn.layers as layers

import datasets.tox21 as tox21


class GCNModel(tf.keras.Model):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.gcn = layers.GraphConv(64, 1)
        
    def call(self, inputs, adjs):
        x = self.gcn(inputs, adjs)
        return x

np.random.seed(0)

def client_data(source, n):
    return source.create_tf_dataset_for_client(source.client_ids[n]).repeat(10).batch(20)

def build_model(adj_shape, features_shape, ):
    features = tf.keras.Input(shape=(12))
    adj = tf.keras.Input(shape=(12))
    

# Wrap a Keras model for use with TFF.
def model_fn(adj_shape=(150, 150), feature_shape=(150, 100)):
    adj = tf.keras.Input(shape=[adj_shape,], sparse=True)
    features = tf.placeholder(tf.int32, shape=(150, 100), name="node"),    
    #features = tf.keras.Input(shape=feature_shape)
    model = GCNModel()
    model(adj, features)
    
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
    #                           kernel_initializer='zeros')
    # ])
    # return tff.learning.from_keras_model(
    #     model,
    #     dummy_batch=sample_batch,
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    

if __name__ == '__main__':
    N_CLIENTS = 2
    # tox21_train, tox21_test = tox21.load_data(n_groups=N_CLIENTS+3)
    # train_data = [client_data(tox21_train, n) for n in range(N_CLIENTS)]

    model_fn()
    # trainer = tff.learning.build_federated_averaging_process(
    #     model_fn,
    #     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    # state = trainer.initialize()
    # for _ in range(5):
    #     state, metrics = trainer.next(state, train_data)
    #     print (metrics.loss)
