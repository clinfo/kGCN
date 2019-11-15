import tensorflow as tf

import layers


def dnn_multitask_model(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
        do_rate = params["do_rate"]
    else:
        is_training = False
        do_rate = 0

    tf.set_random_seed(1)

    net = features["input"]

    for node_num in params["node_nums"]:
        net = tf.layers.dense(net, node_num, activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.dropout(net, rate=do_rate)
    # logits = model_multitask_tox21.multitask_logits(net, labels.shape[1])
    net = tf.layers.dense(net, params["task_num"] * 2)
    logits = tf.reshape(net, [-1, 2, params["task_num"]])
    predictions = {"probabilities": tf.nn.softmax(logits)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "predict_output": tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs
        )

    with tf.variable_scope("loss_fn"):
        each_cost = []
        aucs = []
        accuracies = []
        for task in range(params["task_num"]):
            masked_logits = tf.boolean_mask(
                logits[:, :, task], tf.cast(features["mask_label"][:, task], tf.bool)
            )
            masked_labels = tf.boolean_mask(
                labels[:, task], tf.cast(features["mask_label"][:, task], tf.bool)
            )
            loss_task = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(masked_labels, tf.int32), logits=masked_logits
            )
            loss_task = tf.reduce_sum(loss_task)
            each_cost.append(loss_task)
            prediction = tf.nn.softmax(masked_logits)
            aucs.append(tf.metrics.auc(masked_labels, prediction[:, 1]))
            accuracies.append(
                tf.metrics.accuracy(masked_labels, tf.argmax(prediction, axis=1))
            )
        each_cost = tf.stack(each_cost)
        loss_to_minimize = tf.reduce_sum(each_cost) / tf.reduce_sum(
            features["mask_label"]
        )  # note that mask is not empty. guaranteed.

    auc_keys = ["auc " + task for task in params["task_names"]]
    accuracy_keys = ["accuracy " + task for task in params["task_names"]]
    auc_metrics = dict(zip(auc_keys, aucs))
    accuracy_metrics = dict(zip(accuracy_keys, accuracies))
    metrics = {**accuracy_metrics, **auc_metrics}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss_to_minimize, global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, train_op=train_op
        )  # training_hooks=[logging_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, eval_metric_ops=metrics
        )


def gcn_multitask_model(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
        do_rate = params["do_rate"]
    else:
        is_training = False
        do_rate = 0

    if "use_bias" in params:
        use_bias = params["use_bias"]
    else:
        use_bias = False

    tf.set_random_seed(1)
    # idx = tf.where(tf.not_equal(features['adjs'], 0))
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
    # adj_sparse = tf.SparseTensor(idx, tf.gather_nd(features['adjs'], idx), [128, 50, 50])# features['adjs'].get_shape())
    # new_idx = adj_sparse.indices[:, 1:] + 50 * tf.stack([adj_sparse.indices[:, 0], adj_sparse.indices[:, 0]], 1)
    # adj_sparse_flat = tf.SparseTensor(new_idx, adj_sparse.values, dense_shape=[128 * 50, 128 * 50])
    net = features[
        "features"
    ]  # tf.reshape(features['features'], [-1, features['features'].shape[1]])
    # net = tf.pad(net, [[0, 128 * 50 - int(net.shape[0])], [0, 0]], "constant")
    input_dim = params["node_dim"]
    for i, out_dim in enumerate(params["out_dims"]):
        # net = layers.gcn_layer('graph_conv', net, adj_sparse, input_dim, out_dim, 50, 1, 128)
        w = tf.Variable(
            tf.truncated_normal([input_dim, out_dim], stddev=0.1), dtype=params["dtype"]
        )
        net = tf.reshape(net, [-1, input_dim])
        net = tf.matmul(net, w)
        if use_bias:
            b = tf.Variable(tf.zeros([out_dim]), dtype=params["dtype"])
            net = tf.nn.bias_add(net, b)

        net = tf.reshape(net, [-1, params["max_node_num"], out_dim])
        net = tf.matmul(features["adjs"], net)
        net = tf.nn.relu(net)
        net = tf.reshape(net, [-1, out_dim])
        net = tf.nn.dropout(net, 1.0 - params["do_rate"])
        #        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.reshape(net, [-1, params["max_node_num"], out_dim])
        input_dim = out_dim
        # net = layers.graph_batch_normalization_with_tf('bn', net, features['atom_numbers'], batch_size, 50, input_dim, is_train=istraining)
        # net = layers.dropout_layer(net, do_rate)
    out_dim = params["dense_dim"]
    w = tf.Variable(
        tf.truncated_normal([input_dim, out_dim], stddev=0.1), dtype=params["dtype"]
    )
    b = tf.Variable(tf.zeros([out_dim]), dtype=params["dtype"])
    net = tf.reshape(net, [-1, input_dim])
    net = tf.nn.bias_add(tf.matmul(net, w), b)
    net = tf.reshape(net, [-1, params["max_node_num"], out_dim])
    nodes = tf.reduce_any(tf.cast(features["adjs"], tf.bool), axis=2)
    nodes = tf.cast(nodes, tf.float32)
    net = tf.multiply(tf.transpose(net, perm=[2, 0, 1]), nodes)
    net = tf.transpose(net, perm=[1, 2, 0])
    input_dim = out_dim

    # with tf.variable_scope("bn_3") as scope:
    #   layer=layers.graph_batch_normalization_with_tf("bn", net, enabled_node_nums,batch_size,info.graph_node_num,input_dim,is_train=is_train)
    # layer=layers.graph_batch_normalization("bn",layer,input_dim,info.graph_node_num,init_params_flag=True,params=None)

    with tf.variable_scope("gathering") as scope:
        net = tf.reduce_sum(net, axis=1)
        net = tf.nn.tanh(net)

    if params["mltask"] == "classification":
        net = tf.layers.dense(net, params["task_num"] * 2)
        logits = tf.reshape(net, [-1, 2, params["task_num"]])
        predictions = {"probabilities": tf.nn.softmax(logits, axis=1)}
        if mode == tf.estimator.ModeKeys.PREDICT:
            grad = {
                "grad": tf.gradients(
                    predictions["probabilities"][:, 1],
                    [features["adjs"], features["features"]],
                )
            }
            predictions_new = {**predictions, **grad}
            export_outputs = {
                "predict_output": tf.estimator.export.PredictOutput(predictions_new)
            }
            return tf.estimator.EstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs
            )

        with tf.variable_scope("loss_fn"):
            each_cost = []
            aucs = []
            accuracies = []
            true_positives = []
            true_negatives = []
            false_positives = []
            false_negatives = []
            for task in range(params["task_num"]):
                masked_logits = tf.boolean_mask(
                    logits[:, :, task],
                    tf.cast(features["mask_label"][:, task], tf.bool),
                )
                masked_labels = tf.boolean_mask(
                    labels[:, task], tf.cast(features["mask_label"][:, task], tf.bool)
                )
                loss_task = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(masked_labels, tf.int32), logits=masked_logits
                )
                loss_task = tf.reduce_sum(loss_task)
                each_cost.append(loss_task)
                prediction = tf.nn.softmax(masked_logits)
                aucs.append(tf.metrics.auc(masked_labels, prediction[:, 1]))
                accuracies.append(
                    tf.metrics.accuracy(masked_labels, tf.argmax(prediction, axis=1))
                )
                true_positives.append(
                    tf.metrics.true_positives(
                        masked_labels, tf.argmax(prediction, axis=1)
                    )
                )
                true_negatives.append(
                    tf.metrics.true_negatives(
                        masked_labels, tf.argmax(prediction, axis=1)
                    )
                )
                false_positives.append(
                    tf.metrics.false_positives(
                        masked_labels, tf.argmax(prediction, axis=1)
                    )
                )
                false_negatives.append(
                    tf.metrics.false_negatives(
                        masked_labels, tf.argmax(prediction, axis=1)
                    )
                )
            each_cost = tf.stack(each_cost)
            loss_to_minimize = tf.reduce_sum(each_cost) / tf.reduce_sum(
                features["mask_label"]
            )  # note that mask is not empty. guaranteed.
        auc_keys = ["auc " + task for task in params["task_names"]]
        accuracy_keys = ["accuracy " + task for task in params["task_names"]]
        true_positive_keys = ["true pos " + task for task in params["task_names"]]
        true_negative_keys = ["true neg " + task for task in params["task_names"]]
        false_positive_keys = ["false pos " + task for task in params["task_names"]]
        false_negative_keys = ["false negs " + task for task in params["task_names"]]
        auc_metrics = dict(zip(auc_keys, aucs))
        accuracy_metrics = dict(zip(accuracy_keys, accuracies))
        true_positive_metrics = dict(zip(true_positive_keys, true_positives))
        true_negative_metrics = dict(zip(true_negative_keys, true_negatives))
        false_positive_metrics = dict(zip(false_positive_keys, false_positives))
        false_negative_metrics = dict(zip(false_negative_keys, false_negatives))
        metrics = {
            **accuracy_metrics,
            **auc_metrics,
            **true_positive_metrics,
            **true_negative_metrics,
            **false_positive_metrics,
            **false_negative_metrics,
        }

    elif params["mltask"] == "regression":
        logits = tf.layers.dense(net, params["task_num"])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=logits)

        with tf.variable_scope("loss_fn"):
            each_cost = []
            aucs = []
            accuracies = []
            masked_logits = tf.boolean_mask(
                logits, tf.cast(features["mask_label"], tf.bool)
            )
            masked_labels = tf.boolean_mask(
                labels, tf.cast(features["mask_label"], tf.bool)
            )
            loss_task = tf.sqrt(
                tf.losses.mean_squared_error(labels=labels, predictions=logits)
            )
            prediction = masked_logits
            loss_to_minimize = loss_task
        metrics = {}
    else:
        raise ValueError("{} is not supported.".format(params["mltask"]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss_to_minimize, global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, train_op=train_op
        )  # training_hooks=[logging_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, eval_metric_ops=metrics
        )


def gcn_multitask_model_sparse(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
        do_rate = params["do_rate"]
    else:
        is_training = False
        do_rate = 0

    if "use_bias" in params:
        use_bias = params["use_bias"]
    else:
        use_bias = False

    tf.set_random_seed(1)
    with tf.device("/cpu:0"):
        cumsum = tf.cumsum(features["size"][:, 0])

        adj_row = features["adj_row"].values
        adj_col = features["adj_column"].values
        adj_elem_len = features["adj_elem_len"][:, 0]
        start = tf.cumsum(adj_elem_len, exclusive=True)
        offset = tf.cumsum(features["size"][:, 0], exclusive=True)
        start_size_offset = tf.stack([start, adj_elem_len, offset], 1)

    def offset_index(row_or_column):
        def split_sum(a, x):
            padded_index = tf.concat(
                [
                    tf.zeros(x[0], tf.int64),
                    row_or_column[x[0] : x[0] + x[1]] + x[2],
                    tf.zeros(
                        tf.shape(row_or_column, out_type=tf.int64)[0] - x[0] - x[1],
                        tf.int64,
                    ),
                ],
                0,
            )
            return padded_index

        return split_sum

    with tf.device("/cpu:0"):
        padded_rows = tf.scan(
            offset_index(adj_row), start_size_offset, initializer=adj_row
        )
        padded_columns = tf.scan(
            offset_index(adj_col), start_size_offset, initializer=adj_col
        )
    diagonal_row = tf.reduce_sum(padded_rows, axis=0)
    diagonal_col = tf.reduce_sum(padded_columns, axis=0)
    adj_values = features["adj_values"].values
    adj_shape = [
        tf.reduce_sum(features["size"][:, 0]),
        tf.reduce_sum(features["size"][:, 0]),
    ]
    diagonalized_adj = tf.SparseTensor(
        indices=tf.transpose(tf.stack([diagonal_row, diagonal_col])),
        values=adj_values,
        dense_shape=adj_shape,
    )
    if params["normalize"]:
        degree_hat = tf.sparse.reduce_sum(diagonalized_adj, axis=0)
        diagonalized_adj = diagonalized_adj / tf.sqrt(degree_hat) / tf.expand_dims(tf.sqrt(degree_hat), 1)
        #diagonalized_adj = tf.sparse.expand_dims(diagonalized_adj, 2)
    else:
        adj_degrees = tf.clip_by_value(
        features["adj_degrees"].values, 0, params["max_degree"]
        )  # degree is the number of edges of each node, excluding the one to itself. node with degree 0 is not connected with any other node.
        diagonalized_adj = tf.SparseTensor(
            indices=tf.transpose(tf.stack([diagonal_row, diagonal_col, adj_degrees])),
            values=adj_values,
            dense_shape=adj_shape,
        )
    #diagonalized_adj = tf.sparse_split(
    #    sp_input=diagonalized_adj, num_split=params["max_degree"] + 1, axis=2
    #)
    #diagonalized_adj = [
    #    tf.sparse_reshape(da, adj_shape[:-1]) for da in diagonalized_adj
    #]
    feature_col = features["feature_column"].values
    feature_row = features["feature_row"].values
    feature_elem_len = features["feature_elem_len"][:, 0]
    start_feature = tf.cumsum(feature_elem_len, exclusive=True)
    start_size_offset_feature = tf.stack([start_feature, feature_elem_len, offset], 1)
    with tf.device("/cpu:0"):
        padded_rows_feature = tf.scan(
            offset_index(feature_row),
            start_size_offset_feature,
            initializer=feature_row,
        )
    stacked_row = tf.reduce_sum(padded_rows_feature, axis=0)
    # feature_row = features['feature_row'].values + tf.gather(offset, features['feature_row'].indices[:, 0])
    feature_values = features["feature_values"].values
    net = tf.SparseTensor(
        indices=tf.transpose(tf.stack([stacked_row, feature_col])),
        values=feature_values,
        dense_shape=[tf.reduce_sum(features["size"][:, 0]), features["size"][0, 1]],
    )
    net = tf.sparse_reorder(net)
    net = tf.sparse_tensor_to_dense(net)
    input_dim = params["input_dim"]
    #net = tf.expand_dims(net, 0)
    #net.set_shape([1, None, input_dim])
    for i, out_dim in enumerate(params["out_dims"]):
        # net = layers.BatchGraphConv(out_dim, input_dim=input_dim)([net, diagonalized_adj])
        net = layers.BatchGraphConv(out_dim, params["max_degree"] + 1, input_dim=input_dim, name='graph_conv')(
            [net, diagonalized_adj]
        )
        input_dim = out_dim
        if params["max_pool"]:
            net = layers.GraphMaxPooling(1)(net, diagonalized_adj)
        if params["batch_normalize"]:
            net = tf.keras.layers.BatchNormalization(axis=-1, trainable=is_training)(
                net
            )

    out_dim = params["dense_dim"]
    net = tf.keras.layers.Dense(out_dim, name='graph_dense')(net)

    with tf.device("/cpu:0"):
        start_mols = tf.cumsum(features["size"][:, 0], exclusive=True)
        start_size_mols = tf.stack([start_mols, features["size"][:, 0]], 1)

        def split_sum(a, x):
            mol_gathered = tf.reduce_sum(
                net[x[0] : x[0] + x[1]], axis=0, keepdims=False
            )
            return mol_gathered

        net = tf.scan(split_sum, start_size_mols, initializer=net[0])
    # net = tf.stack([tf.reduce_sum(mol, axis=0, keepdims=True) for mol in split_feature])
    net = tf.nn.tanh(net)
    input_dim = out_dim

    # with tf.variable_scope("bn_3") as scope:
    #   layer=layers.graph_batch_normalization_with_tf("bn", net, enabled_node_nums,batch_size,info.graph_node_num,input_dim,is_train=is_train)
    # layer=layers.graph_batch_normalization("bn",layer,input_dim,info.graph_node_num,init_params_flag=True,params=None)
    if params["mltask"] == "classification":
        if params["multitask"]:
            net = tf.keras.layers.Dense(params['task_num'] * 2)(net)
            logits = tf.reshape(net, [-1, 2, params['task_num']])
        else:
            net = tf.keras.layers.Dense(params["num_classes"])(net)
            logits = tf.reshape(net, [-1, params["num_classes"], 1])
        predictions = {"probabilities": tf.nn.softmax(logits, axis=1)}
        if mode == tf.estimator.ModeKeys.PREDICT:
            #    grad = {'grad': tf.gradients(predictions['probabilities'][:, 1], [features['adjs'], features['features']])}
            #    predictions_new = {**predictions, **grad}
            export_outputs = {
                "predict_output": tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs
            )

        with tf.variable_scope("loss_fn"):
            each_cost = []
            aucs = []
            accuracies = []
            top_30 = []
            top_50 = []
            true_positives = []
            true_negatives = []
            false_positives = []
            false_negatives = []
            ml = features["mask_label"]
            for task in range(params["task_num"]):
                # masked_logits = tf.boolean_mask(logits[:, :, task], tf.cast(ml[:, task], tf.bool))
                # masked_labels = tf.boolean_mask(labels[:, task], tf.cast(features["mask_label"][:, task], tf.bool))
                loss_task = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(labels[:, task], tf.int32), logits=logits[:, :,task]
                )
                loss_task = tf.reduce_sum(loss_task)
                each_cost.append(loss_task)
                prediction = tf.nn.softmax(logits[:,:,task])
                # aucs.append(tf.metrics.auc(labels, prediction[:, 1]))
                accuracies.append(
                    tf.metrics.accuracy(labels, tf.argmax(prediction, axis=1))
                )
                top_30.append(
                    tf.metrics.mean(
                        tf.nn.in_top_k(
                            predictions=prediction,
                            targets=tf.cast(labels[:, task], tf.int32),
                            k=30,
                        )
                    )
                )
                top_50.append(
                    tf.metrics.mean(
                        tf.nn.in_top_k(
                            predictions=prediction,
                            targets=tf.cast(labels[:, task], tf.int32),
                            k=50,
                        )
                    )
                )
                # true_positives.append(tf.metrics.true_positives(labels, tf.argmax(prediction, axis=1)))
                # true_negatives.append(tf.metrics.true_negatives(labels, tf.argmax(prediction, axis=1)))
                # false_positives.append(tf.metrics.false_positives(labels, tf.argmax(prediction, axis=1)))
                # false_negatives.append(tf.metrics.false_negatives(labels, tf.argmax(prediction, axis=1)))
            each_cost = tf.stack(each_cost)
            loss_to_minimize = tf.reduce_sum(each_cost) / tf.cast(tf.reduce_sum(
                features["mask_label"]
            ), tf.float32)  # note that mask is not empty. guaranteed.
        # auc_keys = ['auc_' + task for task in params['task_names']]
        accuracy_keys = ["accuracy_" + task for task in params["task_names"]]
        top30_keys = ["top30_" + task for task in params["task_names"]]
        top50_keys = ["top50_" + task for task in params["task_names"]]
        # true_positive_keys = ['true pos ' + task for task in params['task_names']]
        # true_negative_keys = ['true neg ' + task for task in params['task_names']]
        # false_positive_keys = ['false pos ' + task for task in params['task_names']]
        # false_negative_keys = ['false negs ' + task for task in params['task_names']]
        # auc_metrics = dict(zip(auc_keys, aucs))
        accuracy_metrics = dict(zip(accuracy_keys, accuracies))
        top30_metrics = dict(zip(top30_keys, top_30))
        top50_metrics = dict(zip(top50_keys, top_50))
        # true_positive_metrics = dict(zip(true_positive_keys, true_positives))
        # true_negative_metrics = dict(zip(true_negative_keys, true_negatives))
        # false_positive_metrics = dict(zip(false_positive_keys, false_positives))
        # false_negative_metrics = dict(zip(false_negative_keys, false_negatives))
        # metrics = {**accuracy_metrics, **auc_metrics, **true_positive_metrics, **true_negative_metrics, **false_positive_metrics, **false_negative_metrics}
        metrics = {**accuracy_metrics, **top30_metrics, **top50_metrics}

    elif params["mltask"] == "regression":
        logits = tf.layers.dense(net, params["task_num"])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=logits)

        with tf.variable_scope("loss_fn"):
            each_cost = []
            aucs = []
            accuracies = []
            masked_logits = tf.boolean_mask(
                logits, tf.cast(features["mask_label"], tf.bool)
            )
            masked_labels = tf.boolean_mask(
                labels, tf.cast(features["mask_label"], tf.bool)
            )
            loss_task = tf.sqrt(
                tf.losses.mean_squared_error(labels=labels, predictions=logits)
            )
            prediction = masked_logits
            loss_to_minimize = loss_task
        metrics = {}
    else:
        raise ValueError("{} is not supported.".format(params["mltask"]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss_to_minimize, global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, train_op=train_op
        )  # training_hooks=[logging_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_to_minimize, eval_metric_ops=metrics
        )
