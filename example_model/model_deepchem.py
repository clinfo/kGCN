import tensorflow as tf
import numpy as np
import joblib
import layers

def build_placeholders(info,batch_size=4,adj_channel_num=1,embedding_dim=10):
    placeholders = {
        'adjs':[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)) for a in range(adj_channel_num)] for b in range(batch_size)],
        'nodes': tf.placeholder(tf.int32, shape=(batch_size,info.graph_node_num),name="node"),
        'labels': tf.placeholder(tf.float32, shape=(batch_size,info.label_dim),name="label"),
        'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
        'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
    }
    if info.feature_enabled:
        placeholders['features']=tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.feature_dim),name="feature")
    else:
        placeholders['features']=None
    return  placeholders

def build_model(placeholders,info,batch_size=4,adj_channel_num=1,embedding_dim=10):
    in_adjs=placeholders["adjs"]
    features=placeholders["features"]
    in_nodes=placeholders["nodes"]
    labels=placeholders["labels"]
    mask=placeholders["mask"]
    dropout_rate=placeholders["dropout_rate"]
    wd_b=None
    wd_w=0.1

    layer=features
    input_dim=info.feature_dim
    if features is None:
        layer=emmbeding_layer("embeding",in_nodes,info.all_node_num,embedding_dim,init_params_flag=True,params=None)
        input_dim=embedding_dim
    # layer: batch_size x graph_node_num x dim
    with tf.variable_scope("gcn_1") as scope:
        output_dim=64
        layer = layers.gcn_layer("graph_conv",layer,in_adjs,input_dim,output_dim,
                adj_channel_num=adj_channel_num,node_num=info.graph_node_num,batch_size=batch_size)
        layer = tf.nn.relu(layer)
        input_dim=output_dim
    with tf.variable_scope("pooling_1") as scope:
        layer = layers.graph_max_pooling_layer(layer,in_adjs, input_dim,
                adj_channel_num=adj_channel_num,node_num=info.graph_node_num,batch_size=batch_size)
    with tf.variable_scope("bn_1") as scope:
        layer=layers.graph_batch_normalization("bn",layer,input_dim,info.graph_node_num,init_params_flag=True,params=None)
    with tf.variable_scope("do_1") as scope:
        layer=layers.graph_dropout_layer(layer,info.graph_node_num,input_dim,dropout_rate)

    with tf.variable_scope("gcn_2") as scope:
        output_dim=128
        layer = layers.gcn_layer("graph_conv",layer,in_adjs,input_dim,output_dim,adj_channel_num=adj_channel_num,node_num=info.graph_node_num,batch_size=batch_size)
        layer = tf.sigmoid(layer)
        input_dim=output_dim
    with tf.variable_scope("pooling_2") as scope:
        layer = layers.graph_max_pooling_layer(layer,in_adjs, input_dim,
                adj_channel_num=adj_channel_num,node_num=info.graph_node_num,batch_size=batch_size)
    with tf.variable_scope("bn_2") as scope:
        layer=layers.graph_batch_normalization("bn",layer,input_dim,info.graph_node_num,init_params_flag=True,params=None)
    with tf.variable_scope("do_2") as scope:
        layer=layers.graph_dropout_layer(layer,info.graph_node_num,input_dim,dropout_rate)

    with tf.variable_scope("gcn_3") as scope:
        output_dim=128
        layer = layers.gcn_layer("graph_conv",layer,in_adjs,input_dim,output_dim,adj_channel_num=adj_channel_num,node_num=info.graph_node_num,batch_size=batch_size)
        layer = tf.sigmoid(layer)
        input_dim=output_dim
    with tf.variable_scope("pooling_3") as scope:
        layer = layers.graph_max_pooling_layer(layer,in_adjs, input_dim,
                adj_channel_num=adj_channel_num,node_num=info.graph_node_num,batch_size=batch_size)
    with tf.variable_scope("bn_3") as scope:
        layer=layers.graph_batch_normalization("bn",layer,input_dim,info.graph_node_num,init_params_flag=True,params=None)
    with tf.variable_scope("do_3") as scope:
        layer=layers.graph_dropout_layer(layer,info.graph_node_num,input_dim,dropout_rate)


    with tf.variable_scope("fc4") as scope:
        output_dim=64
        layer = layers.graph_fc_layer("fc",layer,input_dim, output_dim,info.graph_node_num, init_params_flag=True,params=None,wd_w=wd_w,wd_b=wd_b,activate=tf.sigmoid,with_bn=False)
        input_dim=output_dim
    with tf.variable_scope("gathering") as scope:
        layer = layers.graph_gathering_layer(layer)
    with tf.variable_scope("fc5") as scope:
        output_dim=2
        model = layers.fc_layer("fc3",layer,input_dim, output_dim, init_params_flag=True,params=None,wd_w=wd_w,wd_b=wd_b,activate=None,with_bn=False)

    prediction=tf.nn.softmax(model)
    # computing cost and metrics
    cost=mask*tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=model)
    cost_opt=tf.reduce_mean(cost)

    metrics={}
    cost_sum=tf.reduce_sum(cost)

    correct_count=mask*tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)),tf.float32)
    metrics["correct_count"]=tf.reduce_sum(correct_count)
    return model,prediction,cost_opt,cost_sum,metrics

