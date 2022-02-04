import tensorflow as tf

if tf.__version__.split(".")[0] == "2":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
import kgcn.legacy.layers
from kgcn.default_model import DefaultModel


class GCN(DefaultModel):
    def build_placeholders(self, info, config, batch_size, **kwargs):
        # input data types (placeholders) of this neural network
        keys = [
            "adjs",
            "nodes",
            "labels",
            "mask",
            "mask_label",
            "dropout_rate",
            "mask_node",
            "enabled_node_nums",
            "is_train",
            "features",
        ]
        return self.get_placeholders(info, config, batch_size, keys, **kwargs)

    def build_model(self, placeholders, info, config, batch_size, **kwargs):
        adj_channel_num = info.adj_channel_num
        embedding_dim = config["embedding_dim"]
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        in_nodes = placeholders["nodes"]
        labels = placeholders["labels"]
        mask = placeholders["mask"]
        mask_label = placeholders["mask_label"]
        # dropout_rate=placeholders["dropout_rate"]
        dropout_rate = 0.3
        is_train = placeholders["is_train"]
        mask_node = placeholders["mask_node"]
        enabled_node_nums = placeholders["enabled_node_nums"]

        layer = features
        input_dim = info.feature_dim
        with tf.variable_scope("rollout"):
            if features is None:
                layer = K.layers.Embedding(info.all_node_num, embedding_dim)(in_nodes)
                input_dim = embedding_dim
            # layer: batch_size x graph_node_num x dim
            layer = kgcn.legacy.layers.GraphConv(128, adj_channel_num)(
                layer, adj=in_adjs
            )
            layer = kgcn.legacy.layers.GraphBatchNormalization()(
                layer,
                max_node_num=info.graph_node_num,
                enabled_node_nums=enabled_node_nums,
            )
            layer = tf.nn.relu(layer)

            layer = kgcn.legacy.layers.GraphConv(128, adj_channel_num)(
                layer, adj=in_adjs
            )
            layer = kgcn.legacy.layers.GraphBatchNormalization()(
                layer,
                max_node_num=info.graph_node_num,
                enabled_node_nums=enabled_node_nums,
            )
            layer = tf.nn.relu(layer)

            layer = kgcn.legacy.layers.GraphConv(128, adj_channel_num)(
                layer, adj=in_adjs
            )
            layer = kgcn.legacy.layers.GraphBatchNormalization()(
                layer,
                max_node_num=info.graph_node_num,
                enabled_node_nums=enabled_node_nums,
            )
            layer = tf.nn.relu(layer)

            layer = kgcn.legacy.layers.GraphDense(128)(layer)
            layer = tf.nn.relu(layer)

            layer = kgcn.legacy.layers.GraphGather()(layer)
            layer = K.layers.Dense(info.label_dim)(layer)
            prediction = tf.nn.softmax(layer, name="output")
            # computing cost and metrics
            cost = mask * tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=layer
            )
            cost_opt = tf.reduce_mean(cost)
            metrics = {}
            cost_sum = tf.reduce_sum(cost)
            correct_count = mask * tf.cast(
                tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32
            )
            metrics["correct_count"] = tf.reduce_sum(correct_count)
        return layer, prediction, cost_opt, cost_sum, metrics
