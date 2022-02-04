import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.keras as K
else:
    import tensorflow.contrib.keras as K
import numpy as np
import joblib
import kgcn.layers
from tensorflow.python.keras.layers import Dense

def build_placeholders(info,config,batch_size=4,**kwargs):
    adj_channel_num=info.adj_channel_num
    encoder_output_dim=64
    preference_list_length=2
    placeholders = {
        'epsilon': tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,encoder_output_dim),name="epsilon"),
        'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
        'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
        'is_train': tf.placeholder(tf.bool, name="is_train"),
    }
    return  placeholders

def encode(name,inputs,info,batch_size):
    internal_dim=64
    encoder_output_dim=inputs["encoder_output_dim"]
    in_adjs=inputs["adjs"]
    features=inputs["features"]
    dropout_rate=inputs["dropout_rate"]
    is_train=inputs["is_train"]
    enabled_node_nums=inputs['enabled_node_nums']
    adj_channel_num=info.adj_channel_num

    with tf.variable_scope(name):
        layer=features
        layer=kgcn.layers.GraphConv(internal_dim,adj_channel_num)(layer,adj=in_adjs)
        layer=kgcn.layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer = tf.tanh(layer)
        layer=kgcn.layers.GraphConv(internal_dim,adj_channel_num)(layer,adj=in_adjs)
        layer=kgcn.layers.GraphBatchNormalization()(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer = tf.tanh(layer)
        layer=kgcn.layers.GraphDense(internal_dim)(layer)
        layer=tf.sigmoid(layer)
        layer=kgcn.layers.GraphGather()(layer)

        mean_layer=Dense(encoder_output_dim,kernel_initializer='random_uniform')(layer)
        std_layer=Dense(encoder_output_dim)(layer)
        std_layer=tf.nn.softplus(std_layer)
        std_layer=tf.sqrt(std_layer)
        mean_layer=tf.clip_by_value(mean_layer,-100,100)
        std_layer=tf.clip_by_value(std_layer,-5,5)
    return mean_layer,std_layer



def decode_nodes(name,inputs,info):
    dropout_rate=inputs["dropout_rate"]
    layer=inputs["input_layer"]
    input_dim=inputs["input_layer_dim"]
    decoded_output_dim=inputs["output_layer_dim"]
    node_num=inputs["decoded_node_num"]
    is_train=inputs["is_train"]
    enabled_node_nums=inputs['enabled_node_nums']
    with tf.variable_scope(name):
        layer=kgcn.layers.GraphDense(decoded_output_dim,kernel_initializer='random_uniform',name="dense_1")(layer)
    return layer

"""
def decode_links(name,inputs,info):
    dropout_rate=inputs["dropout_rate"]
    internal_dim=64
    layer=inputs["input_layer"]
    input_dim=inputs["input_layer_dim"]
    is_train=inputs["is_train"]
    node_num=inputs["decoded_node_num"]
    enabled_node_nums=inputs['enabled_node_nums']
    with tf.variable_scope(name):
        layer=kgcn.layers.GraphDense(internal_dim,name="dense_1")(layer)
        layer=kgcn.layers.GraphBatchNormalization(name="bn_1")(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer=tf.sigmoid(layer)
        layer=kgcn.layers.GraphDecoderInnerProd()(layer)
    return layer
"""

def decode_links(name,inputs,info):
    dropout_rate=inputs["dropout_rate"]
    internal_dim=64
    layer=inputs["input_layer"]
    input_dim=inputs["input_layer_dim"]
    is_train=inputs["is_train"]
    node_num=inputs["decoded_node_num"]
    enabled_node_nums=inputs['enabled_node_nums']
    with tf.variable_scope(name):
        layer=kgcn.layers.GraphDense(internal_dim,name="dense_1")(layer)
        layer=kgcn.layers.GraphBatchNormalization(name="bn_1")(layer,
            max_node_num=info.graph_node_num,enabled_node_nums=enabled_node_nums)
        layer=tf.sigmoid(layer)
        layer=kgcn.layers.GraphDense(internal_dim,name="dense_2")(layer)
        layer=tf.sigmoid(layer)
        #layer=kgcn.layers.GraphDecoderInnerProd()(layer)
        layer=kgcn.layers.GraphDecoderDistMult()(layer)
    return layer




def print_variables():
    # print variables
    print('## training variables')
    vars_em = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nn")
    for v in vars_em:
        print(v.name)

# TODO: hard coding parameters
def build_model(placeholders,info,config,batch_size=4,**kwargs):
    ##################################################
    ###
    ### dummy encoding
    ###
    ##################################################
    adj_channel_num=info.adj_channel_num
    embedding_dim=64
    encoder_output_dim=64
    adjs=[[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)+"_"+str(p)) for a in range(adj_channel_num)] for b in range(batch_size)] for p in range(2)]
    mask=[tf.placeholder(tf.float32, shape=(batch_size,),name="mask"+"_"+str(p)) for p in range(2)]
    features=[tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.feature_dim),name="feature"+"_"+str(p)) for p in range(2)]
    features=features[0]
    mask=mask[0]
    encoder_output_dim=64
    input_encoder={
        "adjs":adjs[0],
        "features":features,
        "encoder_output_dim":encoder_output_dim,
        "dropout_rate":placeholders["dropout_rate"],
        "is_train":placeholders["is_train"],
        "enabled_node_nums":placeholders['enabled_node_nums'],
        }
    layer_mean,layer_std=encode("encode_nn",input_encoder,info,batch_size=batch_size)
    ##################################################

    # layer_mean: batch_size x dim
    # generating node_num vectors
    layer_std=tf.ones((batch_size,info.graph_node_num,encoder_output_dim))
    layer=layer_std*placeholders["epsilon"] # reparameterization trick
    # layer: batch_size x node_num x dim
    ## decoder
    decoded_node_num=info.graph_node_num
    input_decoder={
        "input_layer":layer,
        "input_layer_dim":64,
        "output_layer_dim":features.shape[2],
        "decoded_node_num":decoded_node_num,
        "dropout_rate":placeholders["dropout_rate"],
        "is_train":placeholders["is_train"],
        "enabled_node_nums":placeholders['enabled_node_nums'],
        }
    decoded_features=decode_nodes("decode_nodes",input_decoder,info)
    ### decoder for links
    decoded_adjs_list=[]
    for c in range(adj_channel_num):
        decoded_adj=decode_links("decode_links_"+str(c),input_decoder,info)
        decoded_adjs_list.append(decoded_adj)
    decoded_adjs=tf.stack(decoded_adjs_list)
    decoded_adjs=tf.transpose(decoded_adjs,[1,0,2,3])
    #
    # sum all costs
    cost=tf.constant(0)
    #cost=mask*(tf.reduce_mean(cost_features,axis=1))
    #cost=mask*(tf.reduce_sum(cost_links,axis=1))
    output=decoded_features

    cost_opt=tf.constant(0)

    cost_sum=tf.constant(0)

    ## TODO: computing metrics
    #correct_count=mask0*tf.reshape(tf.cast(tf.greater(output1,output0),tf.float32),(-1,))
    #metrics["correct_count"]=tf.reduce_sum(correct_count)
    metrics={}
    metrics["correct_count"]=cost_opt # dummy

    ## TODO: prediction (final result)
    metrics["correct_count"]=cost_opt # dummy
    prediction={"feature":tf.sigmoid(decoded_features),"dense_adj":tf.sigmoid(decoded_adjs)}
    return output,prediction,cost_opt,cost_sum,metrics

