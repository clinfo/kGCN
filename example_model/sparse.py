import tensorflow as tf

if tf.__version__.split(".")[0] == "2":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
from kgcn import data_util
from kgcn import layers


def build(config):
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    estimator_config = tf.estimator.RunConfig(
        session_config=session_config, save_checkpoints_steps=config["steps_per_epoch"]
    )
    max_degree = (
        0 if config["normalize_adj_flag"] or not (config["split_adj_flag"]) else 5
    )

    gcn_classifier = tf.estimator.Estimator(
        model_fn=sparse_model,
        model_dir=config["model_dir"],
        params={
            "learning_rate": config["learning_rate"],
            "task_names": config["task_names"],
            "out_dims": [256, 256, 256],
            "dense_dim": 256,
            "batch_normalize": False,
            "max_pool": False,
            "max_degree": max_degree,
            "normalize": config["normalize_adj_flag"],
            "split_adj": config["split_adj_flag"],
            "num_classes": config["num_classes"],
            "input_dim": config["input_dim"],
        },
        config=estimator_config,
    )
    return gcn_classifier


def sparse_model(features, labels, mode, params):
    tf.set_random_seed(1)

    diagonalized_adj, net = data_util.construct_batched_adjacency_and_feature_matrices(
        features["size"][:, 0],
        features["adj_row"].values,
        features["adj_column"].values,
        features["adj_values"].values,
        features["adj_elem_len"][:, 0],
        features["adj_degrees"].values,
        features["feature_row"].values,
        features["feature_column"].values,
        features["feature_values"].values,
        features["feature_elem_len"][:, 0],
        features["size"][0, 1],
        max_degree=params["max_degree"],
        normalize=params["normalize"],
        split_adj=params["split_adj"],
    )

    diagonalized_adj = [diagonalized_adj]
    net.set_shape([None, params["input_dim"]])
    net = tf.expand_dims(net, 0)
    for out_dim in params["out_dims"]:
        net = layers.GraphConv(out_dim, params["max_degree"] + 1)(net, diagonalized_adj)
        if params["max_pool"]:
            net = layers.GraphMaxPooling(params["max_degree"] + 1)(
                net, diagonalized_adj
            )
        if params["batch_normalize"]:
            net = layers.GraphBatchNormalization()(net)
        net = tf.nn.relu(net)

    net = layers.GraphDense(params["dense_dim"])(net)
    net = layers.GraphBatchNormalization()(net)
    net = tf.nn.relu(net)
    net = net[0]

    with tf.device("/cpu:0"):
        start_mols = tf.cumsum(features["size"][:, 0], exclusive=True)
        start_size_mols = tf.stack([start_mols, features["size"][:, 0]], 1)

        def split_sum(a, x):
            mol_gathered = tf.reduce_sum(
                net[x[0] : x[0] + x[1]], axis=0, keepdims=False
            )
            return mol_gathered

        net = tf.scan(split_sum, start_size_mols, initializer=net[0])
    net = tf.nn.tanh(net)

    logits = tf.keras.layers.Dense(params["num_classes"])(net)
    probabilities = tf.nn.softmax(logits)
    predictions = {"probabilities": probabilities}
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "predict_output": tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs
        )

    labels = labels[:, 0]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss_to_minimize = tf.reduce_sum(loss)
    metrics = {
        "accuracy": tf.metrics.accuracy(labels, tf.math.argmax(probabilities, axis=1))
    }
    if params["num_classes"] > 100:
        metrics["top30"] = tf.metrics.mean(
            tf.nn.in_top_k(
                predictions=probabilities, targets=tf.cast(labels, tf.int32), k=30
            )
        )
        metrics["top50"] = tf.metrics.mean(
            tf.nn.in_top_k(
                predictions=probabilities, targets=tf.cast(labels, tf.int32), k=50
            )
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss_to_minimize, global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, train_op=train_op
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, eval_metric_ops=metrics
        )
