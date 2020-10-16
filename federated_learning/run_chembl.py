4#!/usr/bin/env python
#import nest_asyncio
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import optimizers
import tensorflow_federated as tff

import kgcn.layers as layers

from datasets.chembl import load_data

np.random.seed(0)
      
def build_model(max_n_atoms, max_n_types, protein_max_seqlen):
      input_adjs = tf.keras.Input(shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
      input_features = tf.keras.Input(shape=(max_n_atoms, max_n_types), name='features')      
      input_protein_seq = tf.keras.Input(shape=(protein_max_seqlen), name='protein_seq')
      h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
      h = tf.keras.layers.ReLU()(h)
      h = layers.GraphConvFL(64, 1)(h, input_adjs)
      h = tf.keras.layers.ReLU()(h)
      h = layers.GraphGather()(h)
      logits = tf.keras.layers.Dense(2, tf.nn.softmax, input_shape=[64])(h)
      return tf.keras.Model(inputs=[input_adjs, input_features, input_protein_seq], outputs=logits)

def client_data(source, n):
      return source.create_tf_dataset_for_client(source.client_ids[n]).repeat(10).batch(1)

if __name__ == '__main__':
      MAX_N_ATOMS = 150
      MAX_N_TYPES = 100
      PROTEIN_MAX_SEQLEN = 750
      NUM_CLIENTS = 4
      #Load simulation data.
      chembl_train = load_data(MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN, NUM_CLIENTS)

      # # Pick a subset of client devices to participate in training.
      train_data = [client_data(chembl_train, n) for n in range(NUM_CLIENTS)]
      
      # Wrap a Keras model for use with TFF.
      def model_fn():
            model = build_model(MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN)
            return tff.learning.from_keras_model(
                  model,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  input_spec=train_data[0].element_spec)
      
      # Simulate a few rounds of training with the selected client devices.
      trainer = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
      state = trainer.initialize()
      for _ in range(5):
            ## FIXME
            state, metrics = trainer.next(state, train_data)
            print (metrics.loss)
