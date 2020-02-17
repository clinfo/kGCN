import tensorflow as tf
import numpy as np
import joblib
import kgcn.layers
import tensorflow.contrib.keras as K

def build_placeholders(info,config,batch_size=4):
    adj_channel_num=info.adj_channel_num
    embedding_dim=config["embedding_dim"]
    placeholders = {
        'adjs':[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)) for a in range(adj_channel_num)] for b in range(batch_size)],
        'nodes': tf.placeholder(tf.int32, shape=(batch_size,info.graph_node_num),name="node"),
        'labels': tf.placeholder(tf.int64, shape=(batch_size,info.label_dim),name="label"),
        'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
        'mask_label': tf.placeholder(tf.float32, shape=(batch_size,info.label_dim),name="mask_label"),
        'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
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
    in_nodes=placeholders["nodes"]
    labels=placeholders["labels"]
    mask=placeholders["mask"]
    mask_label=placeholders["mask_label"]
    dropout_rate=placeholders["dropout_rate"]
    is_train=placeholders["is_train"]
    enabled_node_nums=placeholders["enabled_node_nums"]
    print("pos_weight:",info.pos_weight)
    print(info.param)
    layer=features
    in_dim=info.feature_dim
    out_dim=info.feature_dim
    #82 -> 128 -> 172
    # layer: batch_size x graph_node_num x dim
    out_dim=128
    layer=kgcn.layers.GraphConv(out_dim,adj_channel_num)(layer,adj=in_adjs)
    layer=kgcn.layers.GraphBatchNormalization()(layer,
        max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums,training=is_train)
    layer=tf.nn.relu(layer)
    layer=K.layers.Dropout(dropout_rate)(layer)
    layer=kgcn.layers.GraphDense(out_dim)(layer)
    layer=tf.nn.relu(layer)
    ###
    out_dim=172
    layer=kgcn.layers.GraphConv(out_dim,adj_channel_num)(layer,adj=in_adjs)
    layer=kgcn.layers.GraphBatchNormalization()(layer,
        max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums,training=is_train)
    layer=tf.nn.relu(layer)
    layer=K.layers.Dropout(dropout_rate)(layer)
    layer=kgcn.layers.GraphDense(out_dim)(layer)
    layer=tf.nn.relu(layer)
    ###
    layer=kgcn.layers.GraphDense(out_dim)(layer)
    layer=kgcn.layers.GraphBatchNormalization()(layer,
        max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums,training=is_train)
    layer=kgcn.layers.GraphGather()(layer)
    layer = tf.nn.tanh(layer)
    
    out_dim=344
    layer=K.layers.Dense(out_dim)(layer)
    layer=K.layers.BatchNormalization()(layer,training=is_train)
    layer=tf.nn.relu(layer)

    logits=K.layers.Dense(info.label_dim)(layer)
    # compute prediction
    predictions = tf.nn.softmax(logits)
    # compute loss
    labels=tf.cast(labels,dtype=tf.float32)
    cw = info['class_weight']
    w = tf.reduce_sum(cw * labels, axis=1)
    unweighted_cost=tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits)
    weighted_cost = unweighted_cost * w
    loss_to_minimize=tf.reduce_sum(weighted_cost)
    # compute correct count
    metrics={}
    correct_count=mask*tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1)),tf.float32)
    metrics["correct_count"]=tf.reduce_sum(correct_count)
    return logits,predictions,loss_to_minimize,loss_to_minimize,metrics
