import tensorflow as tf
import numpy as np
import joblib
import layers
from keras.layers import *
import tensorflow.contrib.keras as K

def build_placeholders(info,config,batch_size=4):
    adj_channel_num=info.adj_channel_num
    placeholders = {
        'adjs':[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)) for a in range(adj_channel_num)] for b in range(batch_size)],
        'nodes': tf.placeholder(tf.int32, shape=(batch_size,info.graph_node_num),name="node"),
        'labels': tf.placeholder(tf.float32, shape=(batch_size,info.label_dim),name="label"),
        'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
        'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
        'sequences': tf.placeholder(tf.int32,shape=(batch_size,info.sequence_max_length),name="sequences"),
        'sequences_len': tf.placeholder(tf.int32,shape=(batch_size,2), name="sequences_len"),
        'is_train': tf.placeholder(tf.bool, name="is_train"),
        'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
    }
    if info.feature_enabled:
        placeholders['features']=tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.feature_dim),name="feature")
    else:
        placeholders['features']=None
    return  placeholders

def build_model(placeholders,info,config,batch_size=4):
    adj_channel_num=info.adj_channel_num
    embedding_dim=config["embedding_dim"]
    in_adjs=placeholders["adjs"]
    features=placeholders["features"]
    sequences=placeholders["sequences"]
    sequences_len=placeholders["sequences_len"]
    in_nodes=placeholders["nodes"]
    labels=placeholders["labels"]
    mask=placeholders["mask"]
    dropout_rate=placeholders["dropout_rate"]
    is_train=placeholders["is_train"]
    enabled_node_nums=placeholders["enabled_node_nums"]

    ###
    ### Graph part
    ###
    with tf.variable_scope("seq_nn") as scope_part:
        layer=features
        input_dim=info.feature_dim
        if features is None:
            layer=K.layers.Embedding(info.all_node_num,embedding_dim)(in_nodes)
            input_dim=embedding_dim
        # layer: batch_size x graph_node_num x dim
        layer=layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.sigmoid(layer)
        layer=layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.sigmoid(layer)
        layer=layers.GraphConv(50,adj_channel_num)(layer,adj=in_adjs)
        layer=layers.GraphMaxPooling(adj_channel_num)(layer,adj=in_adjs)
        layer=layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer=tf.sigmoid(layer)
        layer=K.layers.Dropout(dropout_rate)(layer)
        layer=layers.GraphDense(50)(layer)
        layer=tf.sigmoid(layer)
        layer=layers.GraphGather()(layer)
        graph_output_layer=layer
        graph_output_layer_dim=50

    ###
    ### Sequence part
    ###

    with tf.variable_scope("seq_nn") as scope_part:
        # Embedding
        embedding_dim=10
        layer=K.layers.Embedding(info.sequence_symbol_num,embedding_dim)(sequences)
        # CNN + Pooling
        stride = 4
        layer=K.layers.Conv1D(50,stride,padding="same", activation='relu')(layer)
        layer=K.layers.MaxPooling1D(stride)(layer)
        # LSTM 1
        output_dim=32
        layer=K.layers.LSTM(output_dim,return_sequences=True ,go_backwards=True)(layer)
        # LSTM 2
        layer=K.layers.LSTM(output_dim,return_sequences=False,go_backwards=True)(layer)
            #layer = tf.squeeze(layer)
        seq_output_layer=layer
        seq_output_layer_dim=layer.shape[1]
    ###
    ### Shared part
    ###

    # 32dim (Graph part)+ 32 dim (Sequence part)
    layer=tf.concat([seq_output_layer,graph_output_layer],axis=1)
    input_dim=seq_output_layer_dim+graph_output_layer_dim
    with tf.variable_scope("shared_nn") as scope_part:
        layer=K.layers.Dense(52)(layer)
        layer=K.layers.BatchNormalization()(layer)
        layer=tf.nn.relu(layer)

        layer=K.layers.Dense(info.label_dim)(layer)

    prediction=tf.nn.softmax(layer)
    # computing cost and metrics
    cost=mask*tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=layer)
    cost_opt=tf.reduce_mean(cost)

    metrics={}
    cost_sum=tf.reduce_sum(cost)

    correct_count=mask*tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)),tf.float32)
    metrics["correct_count"]=tf.reduce_sum(correct_count)
    return layer,prediction,cost_opt,cost_sum,metrics

