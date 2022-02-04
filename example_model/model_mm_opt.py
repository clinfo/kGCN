import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
import numpy as np
import kgcn.layers
from kgcn.default_model import DefaultModel

class MultimodalNetwork(DefaultModel):
    def build_placeholders(self,info,config,batch_size,**kwargs):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info,config,batch_size,
            ['adjs','nodes','labels','mask','dropout_rate',
            'enabled_node_nums','is_train','features',
            'sequences','sequences_len','embedded_layer'],**kwargs)
        
    def build_model(self,placeholders,info,config,batch_size,feed_embedded_layer=False,**kwargs):
        self.batch_size = batch_size
        self.input_dim = info.input_dim
        self.adj_channel_num = info.adj_channel_num
        self.sequence_symbol_num = info.sequence_symbol_num
        self.graph_node_num = info.graph_node_num
        self.label_dim = info.label_dim
        self.embedding_dim = config["embedding_dim"]
        self.pos_weight = info.pos_weight
        self.feed_embedded_layer = feed_embedded_layer
        ## aliases
        batch_size = self.batch_size
        in_adjs = self.placeholders["adjs"]
        features = self.placeholders["features"]
        sequences = self.placeholders["sequences"]
        sequences_len = self.placeholders["sequences_len"]
        in_nodes = self.placeholders["nodes"]
        labels = self.placeholders["labels"]
        mask = self.placeholders["mask"]
        dropout_rate = self.placeholders["dropout_rate"]
        is_train = self.placeholders["is_train"]
        enabled_node_nums = self.placeholders["enabled_node_nums"]
        embedded_layer = self.placeholders['embedded_layer']

        print(info.param)
        ## GCN
        layer = features
        print("graph input layer:",layer.shape)
        in_dim=info.feature_dim
        out_dim=info.feature_dim
        with tf.variable_scope("graph_nn") as scope_part:
            # layer: batch_size x graph_node_num x dim
            for i in range(int(info.param["num_gcn_layer"])):
                out_dim=int(in_dim*info.param["layer_dim"+str(i)])
                if out_dim<8:
                    out_dim=8
                enabled_dense=int(info.param["add_dense"+str(i)])
                ###
                layer=kgcn.layers.GraphConv(out_dim,self.adj_channel_num)(layer,adj=in_adjs)
                layer=kgcn.layers.GraphBatchNormalization()(layer,
                    max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
                ###
                layer=tf.nn.relu(layer)
                layer=K.layers.Dropout(dropout_rate)(layer)
                if enabled_dense:
                    layer=kgcn.layers.GraphDense(out_dim)(layer)
                    layer=tf.nn.relu(layer)
                in_dim=out_dim

        layer=kgcn.layers.GraphDense(out_dim)(layer)
        layer=kgcn.layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer=kgcn.layers.GraphGather()(layer)
        layer = tf.nn.tanh(layer)
        graph_output_layer = layer
        graph_output_layer_dim=out_dim
        print("graph output layer:",graph_output_layer.shape)

        with tf.variable_scope("seq_nn") as scope_part:
            # Embedding
            self.embedding_layer = tf.keras.layers.Embedding(self.sequence_symbol_num,self.embedding_dim)(sequences)

            if self.feed_embedded_layer:
                layer = embedded_layer
            else:
                layer = self.embedding_layer
            print("sequence input layer:",layer.shape)
            # CNN + Pooling
            in_dim=self.embedding_dim
            for i in range(int(info.param["num_seq_layer"])):
                out_dim=int(in_dim*info.param["layer_seq_dim"+str(i)])
                if out_dim<8:
                    out_dim=8
                stride = 4
                layer = tf.keras.layers.Conv1D(out_dim, stride, padding="same",
                                                             activation='relu')(layer)
                layer = tf.keras.layers.MaxPooling1D(stride)(layer)
                in_dim=out_dim
            stride = 4
            layer = tf.keras.layers.Conv1D(1, stride,padding="same",
                                                         activation='tanh')(layer)
            layer = tf.squeeze(layer)
            if len(layer.shape) == 1:
                # When batch-size is 1, this shape doesn't have batch-size dimmension due to just previous 'tf.squeeze()'.
                layer = tf.expand_dims(layer, axis=0)
            seq_output_layer = layer
            seq_output_layer_dim = int(layer.shape[1])
            print("sequence output layer:",seq_output_layer.shape)

        layer = tf.concat([seq_output_layer, graph_output_layer], axis=1)
        print("shared_part input:",layer.shape)

        in_dim = seq_output_layer_dim + graph_output_layer_dim
        with tf.variable_scope("shared_nn") as scope_part:
            for i in range(int(info.param["num_dense_layer"])):
                out_dim=int(in_dim*info.param["layer_dense_dim"+str(i)])
                if out_dim<8:
                    out_dim=8
                layer=K.layers.Dense(out_dim)(layer)
                layer=K.layers.BatchNormalization()(layer)
                layer=tf.nn.relu(layer)
                in_dim=out_dim

        layer = tf.keras.layers.Dense(self.label_dim)(layer)

        prediction=tf.nn.softmax(layer)
        # computing cost and metrics
        cost=mask*tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=layer)
        cost_opt=tf.reduce_mean(cost)

        metrics={}
        cost_sum=tf.reduce_sum(cost)

        correct_count=mask*tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)),tf.float32)
        metrics["correct_count"]=tf.reduce_sum(correct_count)
        self.out=layer
        return self, prediction, cost_opt, cost_sum, metrics

    def embedding(self, sess, data):
        key = self.placeholders['sequences']
        feed_dict = {key: data}
        out = sess.run(self.embedding_layer, feed_dict)
        return out

