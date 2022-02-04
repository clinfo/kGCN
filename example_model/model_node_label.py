import tensorflow as tf

if tf.__version__.split(".")[0] == "2":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
import kgcn.layers
from kgcn.default_model import DefaultModel


class GCN(DefaultModel):
    def build_placeholders(self, info, config, batch_size, **kwargs):
        # input data types (placeholders) of this neural network
        keys = [
            "adjs",
            "nodes",
            "mask",
            "dropout_rate",
            "node_label",
            "mask_node_label",
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
        labels = placeholders["node_label"]
        mask_labels = placeholders["mask_node_label"]
        mask = placeholders["mask"]
        enabled_node_nums = placeholders["enabled_node_nums"]
        is_train = placeholders["is_train"]

        layer = features
        input_dim = info.feature_dim
        if features is None:
            layer = K.layers.Embedding(info.all_node_num, embedding_dim)(in_nodes)
            input_dim = embedding_dim
        # layer: batch_size x graph_node_num x dim

        layer = kgcn.layers.GraphConv(64, adj_channel_num)(layer, adj=in_adjs)
        layer = kgcn.layers.GraphBatchNormalization()(
            layer, max_node_num=info.graph_node_num, enabled_node_nums=enabled_node_nums
        )
        layer = tf.nn.relu(layer)

        layer = kgcn.layers.GraphConv(64, adj_channel_num)(layer, adj=in_adjs)
        layer = kgcn.layers.GraphBatchNormalization()(
            layer, max_node_num=info.graph_node_num, enabled_node_nums=enabled_node_nums
        )
        layer = tf.nn.relu(layer)

        layer = kgcn.layers.GraphConv(2, adj_channel_num)(layer, adj=in_adjs)
        prediction = tf.nn.softmax(layer)
        # computing cost and metrics
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=layer)
        cost = mask * tf.reduce_mean(cost, axis=1)
        cost_opt = tf.reduce_mean(cost)

        metrics = {}
        cost_sum = tf.reduce_sum(cost)

        pre_count = tf.cast(
            tf.equal(tf.argmax(prediction, 2), tf.argmax(labels, 2)), tf.float32
        )
        correct_count = mask * tf.reduce_mean(pre_count, axis=1)
        metrics["correct_count"] = tf.reduce_sum(correct_count)
        return layer, prediction, cost_opt, cost_sum, metrics
