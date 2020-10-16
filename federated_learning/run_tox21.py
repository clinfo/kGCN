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
    return source.create_tf_dataset_for_client(n).repeat(3).batch(50)


def build_model():
    input_features = tf.keras.Input(shape=(132, 81), name="features")
    input_adjs = tf.keras.Input(shape=(1, 132, 132), name="adjs", sparse=False)
    input_mask_label = tf.keras.Input(shape=(12), name="mask_label")
    h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    logits = tf.keras.layers.Dense(12, tf.nn.sigmoid, input_shape=[64])(h)
    return keras.Model(
        inputs=[input_features, input_adjs, input_mask_label],
        outputs=tf.stack([logits, input_mask_label]),
    )


class MultitaskBinaryCrossentropyWithMask(keras.losses.Loss):
    def call(self, y_true, model_out):
        logits = model_out[0]
        masks = model_out[1]
        losses = []
        for task in range(12):
            mask = tf.cast(masks[:, task], tf.bool)
            y_true_masked = tf.boolean_mask(y_true[:, task], mask)
            logits_masked = tf.boolean_mask(logits[:, task], mask)
            loss = tf.keras.losses.binary_crossentropy(
                y_true_masked, logits_masked, from_logits=False
            )
            losses.append(loss)
        loss = tf.stack(losses)
        return loss


class AUCMultitask(keras.metrics.AUC):
    def __init__(self, name="auc_multitask", task_number=0, **kwargs):
        super(AUCMultitask, self).__init__(name=name, **kwargs)
        self.task_number = task_number

    def update_state(self, y_true, y_pred, sample_weight=None):
        model_out = y_pred
        logits = model_out[0]
        masks = model_out[1]
        losses = []
        mask = tf.cast(masks[:, self.task_number], tf.bool)
        y_true_masked = tf.boolean_mask(y_true[:, self.task_number], mask)
        logits_masked = tf.boolean_mask(logits[:, self.task_number], mask)
        super(AUCMultitask, self).update_state(y_true_masked, logits_masked)


if __name__ == "__main__":
    config = {
        "normalize_adj_flag": True,
        "with_feature": True,
        "split_adj_flag": False,
        "shuffle_data": False,
        "dataset": "dataset.jbl",
        "validation_data_rate": 0.2,
    }
    _, train_data, valid_data, info = load_and_split_data(
        config,
        filename=config["dataset"],
        valid_data_rate=config["validation_data_rate"],
    )
    adjs = tf.sparse.concat(
        0,
        [
            tf.sparse.SparseTensor(
                train_data["adjs"][i][0][0],
                train_data["adjs"][i][0][1],
                train_data["adjs"][i][0][2],
            )
            for i in range(train_data.num)
        ],
    )
    adjs = tf.sparse.reshape(adjs, [train_data.num, 1, -1, adjs.shape[-1]])
    adjs = tf.sparse.to_dense(adjs)
    labels = train_data["labels"]
    labels[np.isnan(labels)] = 0
    train_dataset = tff.simulation.FromTensorSlicesClientData(
        {
            "bob": (
                OrderedDict(
                    {
                        "features": train_data["features"][:1000],
                        "adjs": adjs[:1000],
                        "mask_label": train_data["mask_label"][:1000],
                    }
                ),
                labels[:1000],
            ),
            "alice": (
                OrderedDict(
                    {
                        "features": train_data["features"][1000:],
                        "adjs": adjs[1000:],
                        "mask_label": train_data["mask_label"][1000:],
                    }
                ),
                labels[1000:],
            ),
        }
    )
    adjs_valid = tf.sparse.concat(
        0,
        [
            tf.sparse.SparseTensor(
                valid_data["adjs"][i][0][0],
                valid_data["adjs"][i][0][1],
                valid_data["adjs"][i][0][2],
            )
            for i in range(valid_data.num)
        ],
    )
    adjs_valid = tf.sparse.reshape(
        adjs_valid, [valid_data.num, 1, -1, adjs_valid.shape[-1]]
    )
    adjs_valid = tf.sparse.to_dense(adjs_valid)
    valid_labels = valid_data["labels"]
    valid_labels[np.isnan(valid_labels)] = 0
    valid_dataset = tff.simulation.FromTensorSlicesClientData(
        {
            "valid": (
                OrderedDict(
                    {
                        "features": valid_data["features"],
                        "adjs": adjs_valid,
                        "mask_label": valid_data["mask_label"],
                    }
                ),
                valid_labels,
            )
        }
    )

    train_data = [client_data(train_dataset, name) for name in ["bob", "alice"]]

    valid_data = [client_data(valid_dataset, "valid")]

    def model_fn():
        model = build_model()
        return tff.learning.from_keras_model(
            model,
            loss=MultitaskBinaryCrossentropyWithMask(),
            input_spec=train_data[0].element_spec,
            metrics=[
                AUCMultitask(name="auc_task" + str(i), task_number=i) for i in range(12)
            ],
        )

    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1),
    )
    state = trainer.initialize()
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    for epoch in range(30):
        for _ in ["bob", "alice"]:
            state, metrics = trainer.next(state, train_data)
            print(f"{epoch:03d} metrics===>\n", metrics)
            test_metrics = evaluation(state.model, valid_data)
            print(f"{epoch:03d} test_metrics\n", test_metrics)
