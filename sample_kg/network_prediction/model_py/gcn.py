import tensorflow as tf
import numpy as np
import joblib
import kgcn.layers
from kgcn.default_model import DefaultModel
import tensorflow.contrib.keras as K


class KGNetwork(DefaultModel):
    def build_placeholders(self,info,config,batch_size,**kwargs):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info,config,batch_size,
            ['adjs','nodes','mask','dropout_rate',
            'enabled_node_nums','is_train','features',
            'preference_label_list'],**kwargs)

    def build_model(self, placeholders, info, config, batch_size=4,feed_embedded_layer=False,**kwargs):
        self.info=info
        self.config=config
        adj_channel_num=info.adj_channel_num
        embedding_dim=config["embedding_dim"]
        in_adjs=placeholders["adjs"]
        features=placeholders["features"]
        in_nodes=placeholders["nodes"]
        label_list=placeholders["preference_label_list"]
        mask=placeholders["mask"]
        enabled_node_nums=placeholders["enabled_node_nums"]
        is_train=placeholders["is_train"]

        layer=features
        input_dim=info.feature_dim
        if layer is None:
            if feed_embedded_layer:
                layer = embedded_layer
            else:
                layer = self._embedding(in_nodes)
            input_dim=embedding_dim
        print(layer)
        # layer: batch_size=1 x graph_node_num x dim
        layer=kgcn.layers.GraphConv(128,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.nn.relu(layer)
        #
        layer=kgcn.layers.GraphConv(128,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.nn.relu(layer)
        #
        prediction=layer
        # computing cost and metrics
        # lp_prediction: batch_size=1 x node_num x embedding_dim
        lp_prediction=tf.matmul(layer,layer,transpose_b=True)
        pred0=tf.gather(prediction[0,:,:],label_list[0,:,0])
        pred1=tf.gather(prediction[0,:,:],label_list[0,:,2])
        pred2=tf.gather(prediction[0,:,:],label_list[0,:,3])
        pred3=tf.gather(prediction[0,:,:],label_list[0,:,5])
        s1=tf.reduce_sum(pred0*pred1,axis=1)
        s2=tf.reduce_sum(pred2*pred3,axis=1)
        #gamma=0.1
        #score = s2 - s1 + gamma
        score = s1 - s2 # add new
        #output=1.0/(1.0+tf.exp(score))
        output = tf.nn.sigmoid(score) # add new
        cost=-1*tf.log(output+1.0e-10)
        self.score = s1
        self.loss = cost
        cost_opt=tf.reduce_mean(cost)
        cost_sum=tf.reduce_sum(cost)
        ####
        # layer: batch_size=1 x node_num x embedding_dim
        #left_pred=distmult_layer.compute_left_prediction(prediction[0,:,:],pred1,label_list[0,:,1])
        ####
        pre_count=tf.cast(tf.greater(s1,s2),tf.float32)
        metrics={}
        metrics["correct_count"]=tf.reduce_sum(pre_count)
        count=tf.shape(label_list[0,:,0])[0]
        metrics["count"]=count
        ###
        self.out=prediction
        #self.left_pred=left_pred
        return self, lp_prediction, cost_opt, cost_sum, metrics

    def _embedding(self, in_nodes):
        embedding_dim=self.config["embedding_dim"]
        layer=K.layers.Embedding(self.info.all_node_num,embedding_dim)(in_nodes)
        return layer

    def embedding(self, data=None):
        key = self.placeholders['nodes']
        feed_dict = {key: data}
        nodes = self.placeholders["nodes"]
        embedd_values = self._embedding(nodes)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(embedd_values, feed_dict)
            return out


