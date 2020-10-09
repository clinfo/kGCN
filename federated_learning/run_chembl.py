#!/usr/bin/env

#import nest_asyncio
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

# Load simulation data.
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

print(len(emnist_train.client_ids))

def client_data(n):
      return source.create_tf_dataset_for_client(source.client_ids[n]).map(
            lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
      ).repeat(10).batch(20)

# Pick a subset of client devices to participate in training.
train_data = [client_data(n) for n in range(3)]

# Grab a single batch of data so that TFF knows what data looks like.
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(train_data[0]).next())

# Wrap a Keras model for use with TFF.
def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
                              kernel_initializer='zeros')
    ])
    return tff.learning.from_keras_model(
        model,
        dummy_batch=sample_batch,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

      # Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
state = trainer.initialize()
for _ in range(5):
    state, metrics = trainer.next(state, train_data)
    print (metrics.loss)
