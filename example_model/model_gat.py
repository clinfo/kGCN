import tensorflow as tf

if tf.__version__.split(".")[0] == "2":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
import kgcn.layers
from kgcn.default_model import DefaultModel


class GAT(DefaultModel):
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
        enabled_node_nums = placeholders["enabled_node_nums"]
        is_train = placeholders["is_train"]
        dropout_rate = placeholders["dropout_rate"]

        layer = features
        # === GIN ===
        block_out = []

        layer = kgcn.layers.GraphDense(50)(layer)
        layer = kgcn.layers.GAT(adj_channel_num)(layer, adj=in_adjs)

        layer = kgcn.layers.GraphDense(50)(layer)
        layer = kgcn.layers.GAT(adj_channel_num)(layer, adj=in_adjs)
        block_out.append(layer)

        layer = kgcn.layers.GraphDense(50)(layer)
        layer = kgcn.layers.GAT(adj_channel_num)(layer, adj=in_adjs)
        block_out.append(layer)

        read_out = [kgcn.layers.GraphGather()(layer) for layer in block_out]
        layer = tf.concat(read_out, axis=1)
        layer = K.layers.Dense(2)(layer)
        # =====

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
        self.out = layer
        return self, prediction, cost_opt, cost_sum, metrics
