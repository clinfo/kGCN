import tensorflow as tf
import numpy as np
import joblib
import kgcn.layers
import tensorflow.contrib.keras as K
from kgcn.default_model import DefaultModel

class MultitaskNetwork(DefaultModel):
    def build_placeholders(self,info,config,batch_size,**kwargs):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info,config,batch_size,
            ['adjs','nodes','labels','mask','dropout_rate',
            'enabled_node_nums','is_train','features','mask_label'])

    def build_model(self,placeholders,info,config,batch_size,feed_embedded_layer=False,**kwargs):
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
        enabled_node_nums=placeholders["enabled_node_nums"]

        layer=features
        with tf.variable_scope("nn"):
            input_dim=info.feature_dim
            # layer: batch_size x graph_node_num x dim
            layer=kgcn.layers.GraphConv(512,adj_channel_num)(layer,adj=in_adjs)
            layer = tf.nn.sigmoid(layer)
            layer=kgcn.layers.GraphGather()(layer)
            layer=K.layers.Dense(info.label_dim*2)(layer)
            ###
            logits = tf.reshape(layer, [-1, 2, info.label_dim])
            predictions = tf.nn.softmax(logits, axis=1)

        metrics,loss_to_minimize=self.compute_metrics(info,logits,labels,mask_label)
        # #data x # task x # class
        predictions=tf.expand_dims(tf.convert_to_tensor(predictions[:,1,:]),-1)
        self.out=logits
        return self,predictions,loss_to_minimize,loss_to_minimize,metrics

    def compute_metrics(self,info,logits,labels,mask_label):
        metrics={}
        aucs = []
        accs = []
        with tf.variable_scope('loss_fn'):
            each_cost = []
            each_correct_count = []
            each_count = []
            for task in range(info.label_dim):
                masked_logits = tf.boolean_mask(logits[:, :, task], tf.cast(mask_label[:, task], tf.bool))
                masked_labels = tf.boolean_mask(labels[:, task], tf.cast(mask_label[:, task], tf.bool))
                loss_task = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(masked_labels, tf.int32), logits=masked_logits)
                loss_task = tf.reduce_sum(loss_task)
                prediction = tf.nn.softmax(masked_logits)
                each_cost.append(loss_task)
                cnt=tf.cast(tf.equal(tf.cast(masked_labels, tf.int64), tf.argmax(prediction, axis=1)),tf.float32)
                each_correct_count.append(tf.cast(tf.reduce_sum(cnt,axis=0),tf.float32))
                each_count.append(tf.shape(cnt)[0])
                aucs.append(tf.metrics.auc(masked_labels, prediction[:, 1]))
                accs.append(tf.metrics.accuracy(masked_labels, tf.argmax(prediction, axis=1)))
            each_cost = tf.stack(each_cost)
            metrics["each_cost"] = each_cost
            metrics["each_correct_count"] = tf.stack(each_correct_count)
            metrics["each_count"] = tf.stack(each_count)
            #metrics["each_count"] = tf.reduce_sum(mask_label,axis=0)
            loss_to_minimize = tf.reduce_sum(each_cost)
            # / (tf.reduce_sum(mask_label)) # note that mask is not empty. guaranteed.
        auc_keys =['auc ' +str(task) for task in range(info.label_dim)]
        auc_metrics = dict(zip(auc_keys, aucs))
        metrics['auc']=auc_metrics
        acc_keys =['acc ' +str(task) for task in range(info.label_dim)]
        acc_metrics = dict(zip(acc_keys, accs))
        metrics['acc']=acc_metrics
        return metrics,loss_to_minimize
