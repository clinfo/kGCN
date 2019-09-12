import tensorflow as tf
import kgcn.layers
from kgcn.default_model import DefaultModel
import tensorflow.contrib.keras as K

class GCN(DefaultModel):
    def build_placeholders(self,info,config,batch_size):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info,config,batch_size,
            ['adjs','nodes','labels','mask','dropout_rate',
            'enabled_node_nums','is_train','features'])

    def build_model(self,placeholders,info,config,batch_size):
        adj_channel_num=info.adj_channel_num
        in_adjs=placeholders["adjs"]
        features=placeholders["features"]
        in_nodes=placeholders["nodes"]
        labels=placeholders["labels"]
        mask=placeholders["mask"]
        enabled_node_nums=placeholders["enabled_node_nums"]
        is_train=placeholders["is_train"]
        dropout_rate=placeholders["dropout_rate"]

        layer=features
        input_dim=info.feature_dim
        layer=kgcn.layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.sigmoid(layer)
        layer=kgcn.layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.sigmoid(layer)
        layer=kgcn.layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
        #layer=layers.GraphMaxPooling(adj_channel_num)(layer,adj=in_adjs)
        layer=kgcn.layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer=tf.sigmoid(layer)
        layer=K.layers.Dropout(dropout_rate)(layer)
        layer=kgcn.layers.GraphDense(50)(layer)
        layer=tf.sigmoid(layer)
        layer=kgcn.layers.GraphGather()(layer)
        layer=K.layers.Dense(2)(layer)
        prediction=tf.nn.softmax(layer,name="output")
        # computing cost and metrics
        cost=mask*tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=layer)
        cost_opt=tf.reduce_mean(cost)

        metrics={}
        cost_sum=tf.reduce_sum(cost)

        correct_count=mask*tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)),tf.float32)
        metrics["correct_count"]=tf.reduce_sum(correct_count)
        return layer,prediction,cost_opt,cost_sum,metrics

