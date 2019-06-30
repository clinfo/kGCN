import tensorflow as tf
import numpy as np
import joblib
import layers
import tensorflow.contrib.keras as K

def build_placeholders(info,config,batch_size=4):
    adj_channel_num=info.adj_channel_num
    placeholders = {
        'adjs':[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)) for a in range(adj_channel_num)] for b in range(batch_size)],
        'nodes': tf.placeholder(tf.int32, shape=(batch_size,info.graph_node_num),name="node"),
        'node_label': tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.label_dim),name="node_label"),
        'mask_node_label': tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.label_dim),name="node_mask_label"),
        'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
        'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
        'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
        'is_train': tf.placeholder(tf.bool, name="is_train"),
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
    in_nodes=placeholders["nodes"]
    labels=placeholders["node_label"]
    mask_labels=placeholders["mask_node_label"]
    mask=placeholders["mask"]
    enabled_node_nums=placeholders["enabled_node_nums"]
    is_train=placeholders["is_train"]

    layer=features
    input_dim=info.feature_dim
    if features is None:
        layer=K.layers.Embedding(info.all_node_num,embedding_dim)(in_nodes)
        input_dim=embedding_dim
    # layer: batch_size x graph_node_num x dim

    layer=layers.GraphConv(64,adj_channel_num)(layer,adj=in_adjs)
    layer=layers.GraphBatchNormalization()(layer,
        max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
    layer=tf.nn.relu(layer)

    layer=layers.GraphConv(64,adj_channel_num)(layer,adj=in_adjs)
    layer=layers.GraphBatchNormalization()(layer,
        max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
    layer=tf.nn.relu(layer)


    layer=layers.GraphConv(2,adj_channel_num)(layer,adj=in_adjs)
    prediction=tf.nn.softmax(layer)
    # computing cost and metrics
    cost=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=layer)
    cost=mask*tf.reduce_mean(cost,axis=1)
    cost_opt=tf.reduce_mean(cost)

    metrics={}
    cost_sum=tf.reduce_sum(cost)

    pre_count=tf.cast(tf.equal(tf.argmax(prediction,2), tf.argmax(labels,2)),tf.float32)
    correct_count=mask*tf.reduce_mean(pre_count,axis=1)
    metrics["correct_count"]=tf.reduce_sum(correct_count)
    return layer,prediction,cost_opt,cost_sum,metrics

