"""
化合物はGraphConvolution
タンパク質はDNNで実装する
profeat使用時に使用する
"""

from kgcn.default_model import DefaultModel
import tensorflow as tf

if tf.__version__.split(".")[0] == "2":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
import kgcn.layers


class GCN(DefaultModel):
    def build_placeholders(self, info, config, batch_size, **kwargs):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(
            info,
            config,
            batch_size,
            [
                "adjs",
                "nodes",
                "labels",
                "mask",
                "dropout_rate",
                "enabled_node_nums",
                "is_train",
                "features",
                "profeat",
            ],
        )

    def build_model(self, placeholders, info, config, batch_size, **kwargs):
        adj_channel_num = info.adj_channel_num
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        in_nodes = placeholders["nodes"]
        labels = placeholders["labels"]
        mask = placeholders["mask"]
        profeat_dim = info.vector_modal_dim[info.vector_modal_name["profeat"]]
        profeat = placeholders["profeat"]
        enabled_node_nums = placeholders["enabled_node_nums"]
        is_train = placeholders["is_train"]
        dropout_rate = placeholders["dropout_rate"]
        profeat = placeholders["profeat"]
        ###
        ### Graph part
        ###
        with tf.variable_scope("graph_nn") as scope_part:
            layer = features
            input_dim = info.feature_dim
            # layer: batch_size x graph_node_num x dim
            layer = kgcn.layers.GraphConv(50, adj_channel_num)(layer, adj=in_adjs)
            layer = tf.sigmoid(layer)
            """
            layer=kgcn.layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
            layer=tf.sigmoid(layer)
            layer=kgcn.layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
            layer=kgcn.layers.GraphMaxPooling(adj_channel_num)(layer,adj=in_adjs)
            layer=kgcn.layers.GraphBatchNormalization()(layer,
                max_node_num=info.graph_node_num,
                enabled_node_nums=enabled_node_nums)
            layer=tf.sigmoid(layer)
            layer=K.layers.Dropout(dropout_rate)(layer)
            """
            layer = kgcn.layers.GraphDense(50)(layer)
            layer = tf.sigmoid(layer)
            layer = kgcn.layers.GraphGather()(layer)
            graph_output_layer = layer
            graph_output_layer_dim = 50

        ###
        ### vector part
        ###
        with tf.variable_scope("profeat_nn") as scope_part:
            layer = profeat
            layer = K.layers.Dense(300)(layer)
            layer = K.layers.BatchNormalization()(layer)
            layer = tf.nn.relu(layer)

            layer = K.layers.Dense(100)(layer)
            layer = K.layers.BatchNormalization()(layer)
            layer = tf.nn.relu(layer)

            layer = K.layers.Dense(64)(layer)
            layer = K.layers.BatchNormalization()(layer)
            layer = tf.nn.relu(layer)

            seq_output_layer = layer
            seq_output_layer_dim = 64

        ###
        ### Shared part
        ###
        # 32dim (Graph part)+ 32 dim (Sequence part)
        layer = tf.concat([seq_output_layer, graph_output_layer], axis=1)
        input_dim = seq_output_layer_dim + graph_output_layer_dim
        with tf.variable_scope("shared_nn") as scope_part:
            layer = K.layers.Dense(52)(layer)
            layer = K.layers.BatchNormalization()(layer)
            layer = tf.nn.relu(layer)

            layer = K.layers.Dense(info.label_dim)(layer)

        prediction = tf.nn.softmax(layer)
        # computing cost and metrics
        cost = mask * tf.nn.softmax_cross_entropy_with_logits(
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
