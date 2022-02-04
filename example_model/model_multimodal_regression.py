import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
import numpy as np
import joblib
import kgcn.layers
from kgcn.default_model import DefaultModel

class GCN(DefaultModel):
    def build_placeholders(self,info,config,batch_size,**kwargs):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info,config,batch_size,
            ['adjs','nodes','labels','mask','mask_label','dropout_rate',
            'enabled_node_nums','is_train','features',
            'dragon'],**kwargs)

    def build_model(self,placeholders,info,config,batch_size,feed_embedded_layer=False,**kwargs):
        adj_channel_num=info.adj_channel_num
        in_adjs=placeholders["adjs"]
        features=placeholders["features"]
        in_nodes=placeholders["nodes"]
        labels=placeholders["labels"]
        mask=placeholders["mask"]
        mask_label=placeholders["mask_label"]
        dropout_rate=placeholders["dropout_rate"]
        is_train=placeholders["is_train"]
        enabled_node_nums=placeholders["enabled_node_nums"]
        dragon_dim=info.vector_modal_dim[info.vector_modal_name["dragon"]]
        dragon = placeholders["dragon"]

        layer=features
        input_dim=info.feature_dim
        # layer: batch_size x graph_node_num x dim
        layer=kgcn.layers.GraphDense(32)(layer)
        layer=kgcn.layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer = tf.nn.relu(layer)

        layer=kgcn.layers.GraphDense(32)(layer)
        layer=kgcn.layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer = tf.nn.relu(layer)

        layer=kgcn.layers.GraphDense(32)(layer)
        layer=kgcn.layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)

        layer=kgcn.layers.GraphGather()(layer)
        graph_layer = tf.nn.tanh(layer)

        layer=dragon
        layer=K.layers.Dense(8)(layer)
        layer=K.layers.BatchNormalization()(layer)
        vec_layer=tf.nn.relu(layer)

        layer=tf.concat([vec_layer,graph_layer],axis=1)

        logits=K.layers.Dense(info.label_dim)(layer)
        # compute prediction
        predictions = logits
        # compute loss
        labels=tf.cast(labels,dtype=tf.float32)
        loss=mask_label*(labels-logits)**2
        loss_to_minimize = tf.reduce_mean(loss)
        loss_sum = tf.reduce_sum(loss)
        # compute correct count
        metrics={}
        metrics["error_sum"]=loss_sum
        metrics["count"]=tf.reduce_sum(mask_label)
        return logits,predictions,loss_to_minimize,loss_sum,metrics

