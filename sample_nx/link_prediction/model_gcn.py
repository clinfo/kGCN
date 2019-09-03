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
        'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
        'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
        'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
        'is_train': tf.placeholder(tf.bool, name="is_train"),
    }

    placeholders['preference_label_list']= tf.placeholder(tf.int64, shape=(batch_size,None,6),name="preference_label_list")
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
    label_list=placeholders["preference_label_list"]
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
    layer=tf.nn.sigmoid(layer)
    #
    prediction=layer
    lp_prediction=tf.matmul(layer,layer,transpose_b=True)
    pred0=tf.gather(prediction[0,:,:],label_list[0,:,0])
    pred1=tf.gather(prediction[0,:,:],label_list[0,:,2])
    pred2=tf.gather(prediction[0,:,:],label_list[0,:,3])
    pred3=tf.gather(prediction[0,:,:],label_list[0,:,5])
    s1=tf.reduce_sum(pred0*pred1,axis=1)
    s2=tf.reduce_sum(pred2*pred3,axis=1)
    output=1.0/(1.0+tf.exp(s2-s1))
    cost=-1*tf.log(output+1.0e-10)
    # computing cost and metrics
    #cost=mask*tf.reduce_mean(cost,axis=1)
    print(cost)
    cost_opt=tf.reduce_mean(cost)
    metrics={}
    cost_sum=tf.reduce_sum(cost)
    ####
    pre_count=tf.cast(tf.greater(s1,s2),tf.float32)
    metrics["correct_count"]=tf.reduce_sum(pre_count)
    ###
    count=tf.shape(label_list[0,:,0])[0]
    metrics["count"]=count
    return layer,lp_prediction,cost_opt,cost_sum,metrics

