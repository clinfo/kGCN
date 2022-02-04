import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
from kgcn import layers
from kgcn.default_model import DefaultModel


class GCN(DefaultModel):
    def build_placeholders(self, info, config, batch_size, **kwargs):
        keys = ['adjs', 'nodes', 'labels', 'mask', 'mask_label', 'mask_node', 'dropout_rate', 'is_train',
                'enabled_node_nums', 'features']
        return self.get_placeholders(info, config, batch_size, keys, **kwargs)

    def build_model(self, placeholders, info, config, batch_size, **kwargs):
        adj_channel_num = info.adj_channel_num
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        in_nodes = placeholders["nodes"]
        labels = placeholders["labels"]
        mask = placeholders["mask"]
        dropout_rate = placeholders['dropout_rate']
        enabled_node_nums = placeholders["enabled_node_nums"]

        layer = features
        # layer: batch_size x graph_node_num x dim
        layer = layers.GraphConv(256, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.relu(layer)

        layer = layers.GraphConv(256, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.relu(layer)

        layer = layers.GraphConv(256, adj_channel_num)(layer, adj=in_adjs)
        layer = tf.nn.relu(layer)

        layer = layers.GraphDense(256)(layer)
        layer = layers.GraphBatchNormalization()(layer, max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer = tf.nn.relu(layer)

        layer = layers.GraphGather()(layer)
        layer = tf.nn.tanh(layer)
        layer = K.layers.Dense(info.label_dim)(layer)
        prediction = tf.nn.softmax(layer, name="output")
        # computing cost and metrics
        cost = mask * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=layer)
        cost_opt = tf.reduce_mean(cost)
        metrics = {}
        cost_sum = tf.reduce_sum(cost)
        correct_count = mask * tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32)
        metrics["correct_count"] = tf.reduce_sum(correct_count)
        self.out = layer
        return self, prediction, cost_opt, cost_sum, metrics
