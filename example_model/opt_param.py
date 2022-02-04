import tensorflow as tf
import kgcn.layers
import tensorflow.contrib.keras as K
from kgcn.default_model import DefaultModel


class GCN(DefaultModel):
    def build_placeholders(self, info, config, batch_size, **kwargs):
        # input data types (placeholders) of this neural network
        keys = [
            "adjs",
            "nodes",
            "labels",
            "mask",
            "dropout_rate",
            "enabled_node_nums",
            "is_train",
            "mask_node",
            "features",
        ]
        return self.get_placeholders(info, config, batch_size, keys, **kwargs)

    def build_model(self, placeholders, info, config, batch_size, **kwargs):
        adj_channel_num = info.adj_channel_num
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        in_nodes = placeholders["nodes"]
        labels = placeholders["labels"]
        mask = placeholders["mask"]
        dropout_rate = placeholders["dropout_rate"]
        is_train = placeholders["is_train"]
        mask_node = placeholders["mask_node"]
        enabled_node_nums = placeholders["enabled_node_nums"]
        internal_dim = 100
        #
        layer = features
        input_dim = info.feature_dim
        print(info.param["num_gcn_layer"])
        for i in range(int(info.param["num_gcn_layer"])):
            layer = kgcn.layers.GraphConv(internal_dim, adj_channel_num)(
                layer, adj=in_adjs
            )
            layer = kgcn.layers.GraphBatchNormalization()(
                layer,
                max_node_num=info.graph_node_num,
                enabled_node_nums=enabled_node_nums,
            )
            layer = tf.sigmoid(layer)
            layer = K.layers.Dropout(dropout_rate)(layer)
        layer = kgcn.layers.GraphDense(internal_dim)(layer)
        layer = tf.sigmoid(layer)
        layer = kgcn.layers.GraphGather()(layer)
        output_dim = 2
        layer = K.layers.Dense(output_dim)(layer)
        prediction = tf.nn.softmax(layer)

        # computing cost and metrics
        cost = mask * tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.stop_gradient(labels), logits=layer
        )
        cost_opt = tf.reduce_mean(input_tensor=cost)

        metrics = {}
        cost_sum = tf.reduce_sum(input_tensor=cost)

        correct_count = mask * tf.cast(
            tf.equal(tf.argmax(input=prediction, axis=1), tf.argmax(input=labels, axis=1)), tf.float32
        )
        metrics["correct_count"] = tf.reduce_sum(input_tensor=correct_count)
        return layer, prediction, cost_opt, cost_sum, metrics
