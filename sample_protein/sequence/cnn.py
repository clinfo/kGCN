import tensorflow as tf
import numpy as np
import joblib
import kgcn.layers
import tensorflow.keras.layers as klayer
import tensorflow.contrib.keras as K
from kgcn.default_model import DefaultModel

class SeqCNN(DefaultModel):
    def build_placeholders(self,info,config,batch_size,**kwargs):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info,config,batch_size,
            ['labels','dropout_rate',
            'is_train',
            'sequences','sequences_len','embedded_layer'],**kwargs)
        
    def build_model(self,placeholders,info,config,batch_size,feed_embedded_layer=False,**kwargs):
        self.batch_size = batch_size
        self.input_dim = info.input_dim
        self.adj_channel_num = info.adj_channel_num
        self.sequence_symbol_num = info.sequence_symbol_num
        self.graph_node_num = info.graph_node_num
        self.label_dim = info.label_dim
        self.embedding_dim = config["embedding_dim"]
        self.class_weight = info.class_weight
        self.feed_embedded_layer = feed_embedded_layer
        ## aliases
        batch_size = self.batch_size
        sequences = self.placeholders["sequences"]
        sequences_len = self.placeholders["sequences_len"]
        labels = self.placeholders["labels"]
        dropout_rate = self.placeholders["dropout_rate"]
        is_train = self.placeholders["is_train"]
        embedded_layer = self.placeholders['embedded_layer']

        # Embedding
        self.embedding_layer = tf.keras.layers.Embedding(self.sequence_symbol_num,self.embedding_dim)(sequences)

        if self.feed_embedded_layer:
            layer = embedded_layer
        else:
            layer = self.embedding_layer
        print("sequence input layer:",layer.shape)
        # CNN + Pooling
        stride = 4
        layer = tf.keras.layers.Conv1D(505, stride, padding="same",
                                                     activation='relu')(layer)
        layer = tf.keras.layers.MaxPooling1D(stride)(layer)

        stride = 3
        layer = tf.keras.layers.Conv1D(200, stride, padding="same",
                                                     activation='relu')(layer)
        layer = tf.keras.layers.MaxPooling1D(stride)(layer)

        stride = 2
        layer = tf.keras.layers.Conv1D(100, stride, padding="same",
                                                     activation='relu')(layer)
        layer = tf.keras.layers.MaxPooling1D(stride)(layer)

        layer = tf.keras.layers.Conv1D(1, stride,padding="same",
                                                     activation='tanh')(layer)
        layer = tf.squeeze(layer)

        if len(layer.shape) == 1:
            # When batch-size is 1, this shape doesn't have batch-size dimmension due to just previous 'tf.squeeze()'.
            layer = tf.expand_dims(layer, axis=0)
        seq_output_layer = layer
        seq_output_layer_dim = layer.shape[1]
        print("sequence output layer:",seq_output_layer.shape)

        layer = seq_output_layer

        with tf.variable_scope("shared_nn") as scope_part:
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.Dense(52)(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.nn.relu(layer)

        layer = tf.keras.layers.Dense(self.label_dim)(layer)

        prediction=tf.nn.softmax(layer)
        #prediction=tf.reshape(prediction,(-1,1,self.label_dim))
        # computing cost and metrics
        cost=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=layer)
        w=labels*self.class_weight
        print(tf.reduce_sum(w,axis=1))
        cost_opt=tf.reduce_mean(cost*w)

        metrics={}
        cost_sum=tf.reduce_sum(cost)

        correct_count=tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)),tf.float32)
        metrics["correct_count"]=tf.reduce_sum(correct_count)
        self.out=layer
        return self, prediction, cost_opt, cost_sum, metrics

    def embedding(self, sess, data):
        key = self.placeholders['sequences']
        feed_dict = {key: data}
        out = sess.run(self.embedding_layer, feed_dict)
        return out

