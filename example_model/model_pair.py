import tensorflow as tf
import numpy as np
import joblib
import layers
import tensorflow.contrib.keras as K

def build_placeholders(info,config,batch_size=4):
    adj_channel_num=info.adj_channel_num
    preference_list_length=2
    adjs=[[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)+"_"+str(p)) for a in range(adj_channel_num)] for b in range(batch_size)] for p in range(preference_list_length)]
    placeholders = {
        'adjs':adjs,
        'nodes': [tf.placeholder(tf.int32, shape=(batch_size,info.graph_node_num),name="node"+"_"+str(p)) for p in range(preference_list_length)],
        'labels': [tf.placeholder(tf.float32, shape=(batch_size,info.label_dim),name="label"+"_"+str(p)) for p in range(preference_list_length)],
        'mask': [tf.placeholder(tf.float32, shape=(batch_size,),name="mask"+"_"+str(p)) for p in range(preference_list_length)],
        'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
    }
    if info.feature_enabled:
        placeholders['features']=[tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.feature_dim),name="feature"+"_"+str(p)) for p in range(preference_list_length)]
    else:
        placeholders['features']=None
    return  placeholders

def build_nn(inputs,info,config,batch_size=4):
    adj_channel_num=info.adj_channel_num
    preference_list_length=2
    internal_dim=32
    in_adjs=inputs["adjs"]
    features=inputs["features"]
    in_nodes=inputs["nodes"]
    dropout_rate=inputs["dropout_rate"]

    layer=features
    input_dim=info.feature_dim
    if features is None:
        layer=K.layers.Embedding(info.all_node_num,embedding_dim)(in_nodes)
        input_dim=embedding_dim
    # layer: batch_size x graph_node_num x dim
    layer=layers.GraphConv(internal_dim,adj_channel_num)(layer,adj=in_adjs)
    layer=tf.sigmoid(layer)
    layer=layers.GraphConv(internal_dim,adj_channel_num)(layer,adj=in_adjs)
    layer=tf.sigmoid(layer)
    layer=layers.GraphConv(internal_dim,adj_channel_num)(layer,adj=in_adjs)
    layer=layers.GraphMaxPooling(adj_channel_num)(layer,adj=in_adjs)
    layer=layers.GraphBatchNormalization()(layer,
        max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
    layer=tf.sigmoid(layer)
    layer=K.layers.Dropout(dropout_rate)(layer)
    layer=layers.GraphDense(internal_dim)(layer)
    layer=tf.sigmoid(layer)
    layer=layers.GraphGather()(layer)
    layer=K.layers.Dense(info.label_dim)(layer)
    return model

def print_variables():
    # print variables
    print('## training variables')
    vars_em = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nn")
    for v in vars_em:
        print(v.name)

def build_model(placeholders,info,config,batch_size=4):
    ## compute output 0
    if info.feature_enabled:
        features=placeholders["features"][0]
    else:
        features=placeholders["features"]
    inputs0={
        "adjs":placeholders["adjs"][0],
        "nodes":placeholders["nodes"][0],
        "dropout_rate":placeholders["dropout_rate"],
        "features":features}
    mask0=placeholders["mask"][0]
    labels0=placeholders["labels"][0]

    output0=build_nn(inputs0,info,config,batch_size=batch_size)
    ## compute output 1
    if info.feature_enabled:
        features=placeholders["features"][1]
    else:
        features=placeholders["features"]
    inputs1={
        "adjs":placeholders["adjs"][1],
        "nodes":placeholders["nodes"][1],
        "dropout_rate":placeholders["dropout_rate"],
        "features":features}
    mask1=placeholders["mask"][1]
    labels1=placeholders["labels"][1]

    output1=build_nn(inputs1,info,config,batch_size=batch_size)
    #

    # RankNet cost
    output=1.0/(1.0+tf.exp(output0-output1))
    cost=-1*mask0*tf.log(output+1.0e-10)
    #

    # Another cost function
    #c=0.3
    #output=1.0/(1.0+tf.exp(tf.nn.relu(output0-output1+c)-c))
    #output=tf.reshape(output,(-1,))
    #cost=-1*mask0*tf.log(output+1.0e-10)

    # Not work
    #c=0.3
    #output=tf.nn.relu(tf.nn.sigmoid(output0)-tf.nn.sigmoid(output1)+c)
    #output=tf.reshape(output,(-1,))
    #cost=mask0*tf.exp(output+1.0e-10)

    prediction=[output0,output1]

    cost_opt=tf.reduce_mean(cost)
    cost_sum=tf.reduce_sum(cost)
    metrics={}
    correct_count=mask0*tf.reshape(tf.cast(tf.greater(output1,output0),tf.float32),(-1,))
    metrics["correct_count"]=tf.reduce_sum(correct_count)
    miss_count=mask0*tf.reshape(tf.cast(tf.less(output1,output0),tf.float32),(-1,))
    metrics["miss_count"]=tf.reduce_sum(miss_count)

    print_variables()
    return output,prediction,cost_opt,cost_sum,metrics

