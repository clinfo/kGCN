from kgcn.default_model import DefaultModel
import tensorflow as tf
import kgcn.layers
import tensorflow.contrib.keras as K

class GCN(DefaultModel):
    def build_placeholders(self,info,config,batch_size):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info,config,batch_size,
            ['adjs','nodes','labels','mask','dropout_rate',
            'enabled_node_nums','is_train','features',
            'mask_label','mask_node'])

    def build_model(placeholders,info,config,batch_size=4):
        adj_channel_num=info.adj_channel_num
        embedding_dim=config["embedding_dim"]
        in_adjs=placeholders["adjs"]
        features=placeholders["features"]
        in_nodes=placeholders["nodes"]
        labels=placeholders["labels"]
        mask=placeholders["mask"]
        mask_label=placeholders["mask_label"]
        dropout_rate=placeholders["dropout_rate"]
        is_train=placeholders["is_train"]
        mask_node=placeholders["mask_node"]
        enabled_node_nums=placeholders["enabled_node_nums"]

        layer=features
        input_dim=info.feature_dim
        if features is None:
            layer=K.layers.Embedding(info.all_node_num,embedding_dim)(in_nodes)
            input_dim=embedding_dim
        # layer: batch_size x graph_node_num x dim
        layer=layers.GraphConv(256,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.sigmoid(layer)
        layer=layers.GraphConv(256,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.sigmoid(layer)
        layer=layers.GraphDense(256)(layer)
        layer=tf.sigmoid(layer)
        layer=layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
        #layer=layers.GraphMaxPooling(adj_channel_num)(layer,adj=in_adjs)
        layer=layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer=tf.sigmoid(layer)
        layer=layers.GraphDense(50)(layer)
        layer=tf.sigmoid(layer)
        layer=layers.GraphGather()(layer)
        layer=K.layers.Dense(info.label_dim)(layer)
        ###
        ### multi-task loss
        ###
        prediction=tf.sigmoid(layer)
        # computing cost and metrics
        # cost (batch_size x labels) => batch_size
        if "pos_weight" in info:
            cost = mask * tf.reduce_sum(mask_label*tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=layer, pos_weight=info.pos_weight), axis=1)
        else:
            cost = mask * tf.reduce_sum(mask_label*tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=layer),axis=1)

        cost_opt=tf.reduce_mean(cost)

        metrics={}
        cost_sum=tf.reduce_sum(cost)

        def binary_activation(x,thresh):
            cond = tf.less(x, tf.ones(tf.shape(x))*thresh)
            out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
            return out

        correct_count=mask*tf.cast(tf.reduce_all(tf.equal(binary_activation(prediction,0.5), labels),axis=1),tf.float32)
        metrics["correct_count"]=tf.reduce_sum(correct_count)
        return layer,prediction,cost_opt,cost_sum,metrics

