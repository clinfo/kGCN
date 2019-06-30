#
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np
#import hyopt as hy
#FLAGS = tf.app.flags.FLAGS

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        #dtype = tf.half if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape,initializer_name, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    #dtype = tf.half if FLAGS.use_fp16 else tf.float32
    dtype = tf.float32
    if initializer_name=="normal":
        stddev=1e-1
        var = _variable_on_cpu(name,shape,
                tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    elif initializer_name=="zero":
        var = _variable_on_cpu(name, shape,
            tf.constant_initializer(0.0,dtype=dtype))
    elif initializer_name=="xavier":
        var = _variable_on_cpu(name, shape,
            tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)* wd
        tf.add_to_collection('losses', weight_decay)
    return var

def _normalization(data,name):
    bias=1.0
    alpha=0.001 / 9.0
    beta=0.75
    output = tf.nn.lrn(data, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    return output

###
### for graph convolution net
###

def embedding_layer(name,input_data,dim_dict,dim_features,wd_w=0.1,init_params_flag=True,params=None):
    if not init_params_flag:
        tf.get_variable_scope().reuse_variables()
    w = _variable_with_weight_decay('weights', [dim_dict, dim_features],
        initializer_name="normal", wd=wd_w)
    return tf.gather(w,input_data)

def gcn_layer(name,features,adjs, dim_in,dim_out,node_num,adj_channel_num,batch_size,wd_w=0.1,wd_b=None,init_params_flag=True,params=None):
    if not init_params_flag:
        tf.get_variable_scope().reuse_variables()
    ws=[]
    bs=[]
    for adj_ch in range(adj_channel_num):
        w = _variable_with_weight_decay('weights'+str(adj_ch), [dim_in, dim_out],
            initializer_name="normal", wd=wd_w)
        b = _variable_with_weight_decay('biases'+str(adj_ch), [dim_out],
            initializer_name="zero",wd=wd_b)
        ws.append(w)
        bs.append(b)

    o=[[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
    for batch_idx in range(batch_size):
        for adj_ch in range(adj_channel_num):
            adj=adjs[batch_idx][adj_ch]
            fb=features[batch_idx,:,:]
            fw=tf.matmul(fb,ws[adj_ch])+bs[adj_ch]
            el=tf.sparse_tensor_dense_matmul(adj,fw)
            o[batch_idx][adj_ch]=el
        o[batch_idx] = tf.add_n(o[batch_idx])
    return  tf.stack(o)

def gcn_res_layer(name,features,adjs, dim_in,dim_out,node_num,adj_channel_num,batch_size,activate, init_params_flag=True,params=None):
    if not init_params_flag:
        tf.get_variable_scope().reuse_variables()
    temp_layer=features
    layer = gcn_layer(name,temp_layer,adjs,dim_in,dim_out,
            node_num=node_num,adj_channel_num=adj_channel_num,batch_size=batch_size,
            init_params_flag=init_params_flag)
    layer = activate(layer)
    with tf.variable_scope("linear") as scope:
        layer += gcn_layer(name,temp_layer,adjs,dim_in,dim_out,
                node_num=node_num,adj_channel_num=adj_channel_num,batch_size=batch_size,
                init_params_flag=init_params_flag)
    return layer


def graph_max_pooling_layer(features,adjs, dim,node_num,adj_channel_num,batch_size, init_params_flag=True,params=None):
    o=[[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
    for batch_idx in range(batch_size):
        for adj_ch in range(adj_channel_num):
            vec=[None for _ in range(dim)]
            for k in range(dim):
                adj=adjs[batch_idx][adj_ch]
                fb=features[batch_idx,:,k]
                x=adj*fb
                #d=tf.sparse_to_dense(x,(node_num,node_num),0.0)
                d=tf.sparse_tensor_to_dense(x)
                el=tf.reduce_max(d,axis=1)
                vec[k]=el
            o[batch_idx][adj_ch]=tf.stack(vec, axis=1)
        o[batch_idx] = tf.add_n(o[batch_idx])
    return  tf.stack(o)

def graph_gathering_layer(features):
    # features: batch_size x graph_num x feature_size
    return tf.reduce_sum(features,axis=1)

###
###
###
def graph_batch_normalization(name,data,input_dim,node_num,init_params_flag=True,params=None,is_train=True):
    layer=tf.reshape(data,(-1, node_num*input_dim))
    layer=batch_normalization(name,layer,node_num*input_dim,init_params_flag,is_train=is_train)
    return tf.reshape(layer,(-1, node_num, input_dim))

def graph_batch_normalization_with_tf(name,data,enabled_node_nums,batch_size,node_num,input_dim,is_train=True):
    data.set_shape([batch_size,node_num,input_dim]) #shape needs to be explicitly specified. tf cannot automatically keep track of the size.
    extracted_nodes=[feature_map[:enabled_node_num] for feature_map, enabled_node_num in zip(tf.unstack(data), tf.unstack(enabled_node_nums))]
    stacked_nodes=tf.concat(extracted_nodes,0)
    normalized_data=tf.layers.batch_normalization(stacked_nodes,training=is_train)
    split_data=tf.split(normalized_data,enabled_node_nums)
    padded_data=[tf.pad(feature_map, [[0, node_num-tf.shape(feature_map)[0]], [0, 0]]) for feature_map in split_data]
    output = tf.stack(padded_data)
    output.set_shape([batch_size,node_num,input_dim]) # shape needs to be explicitly specified again. tf cannot figure it out.
    return output

def graph_batch_normalization_(name,data,
        mask_node,
        input_dim,node_num,
        adj_index=0,
        init_params_flag=True,params=None,is_train=True):
    #layer=tf.reshape(data,(-1, node_num,input_dim))
    layer=data
    mask_node=mask_node[:,0,:]
    mask_input=tf.expand_dims(mask_node, -1)
    mask_input=tf.tile(mask_input,[1, 1, input_dim])
    layer=layer*mask_input
    #layer=batch_normalization(name,layer,node_num*input_dim,init_params_flag,is_train=is_train)
    if not init_params_flag:
        tf.get_variable_scope().reuse_variables()
    gamma = _variable_with_weight_decay(
        'gamma',
        shape=[input_dim],
        initializer_name="normal",
        wd=None)
    beta  = _variable_with_weight_decay(
        'beta',
        shape=[input_dim],
        initializer_name="normal",
        wd=None)

    layer=tf.reshape(layer,(-1,input_dim))
    mask_layer=tf.reshape(mask_input,(-1,input_dim))
    eps = 1e-5

    #mask_node: batch_size x node_num
    num=tf.reduce_sum(mask_node)
    mean=tf.reduce_sum(layer,axis=0)/num
    var_temp=(layer-(mask_layer*mean))**2
    variance =tf.reduce_sum(var_temp,axis=0)/num
    layer_norm=(layer - mean) / tf.sqrt(variance + eps)
    layer=tf.where(is_train,layer_norm,layer)
    layer=gamma * layer + beta

    return tf.reshape(layer,(-1, node_num, input_dim))


def batch_normalization(name,data,dim,init_params_flag=True,params=None,is_train=True):
    if not init_params_flag:
        tf.get_variable_scope().reuse_variables()
    gamma = _variable_with_weight_decay(
        'gamma',
        shape=[dim],
        initializer_name="normal",
        wd=None)
    beta  = _variable_with_weight_decay(
        'beta',
        shape=[dim],
        initializer_name="normal",
        wd=None)

    eps = 1e-5
    mean, variance = tf.nn.moments(data, [0])
    data=(data - mean) / tf.sqrt(variance + eps)
    return gamma * data + beta


def lstm_layer(x,n_steps,n_output,init_params_flag=True):
    if not init_params_flag:
        tf.get_variable_scope().reuse_variables()
    # x: (batch_size, n_steps, n_input)

    # a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, axis=1)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.LSTMCell(
            n_output, forget_bias=1.0,activation=tf.tanh,
            initializer=tf.constant_initializer(0.0))
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # 'outputs' is a list of output tensor of shape (batch_size, n_output)
    # and change back dimension to (batch_size, n_step, n_output)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs

def graph_fc_layer(name,data,dim_in,dim_out,node_num,wd_w=0.1,wd_b=None,activate=None,with_bn=False,init_params_flag=True,params=None,is_train=True):
    if with_bn:
        layer=tf.reshape(data,(-1,dim_in))
        layer=fc_layer(name,layer,dim_in, dim_out,wd_w,wd_b,activate=None,with_bn=False,init_params_flag=init_params_flag,is_train=is_train)
        layer=tf.reshape(layer,(-1,node_num,dim_out))
        layer=graph_batch_normalization(name+"/bn",layer,dim_out,node_num,init_params_flag,is_train=is_train)
        if activate is None:
            return layer
        else:
            return activate(layer)
    else:
        layer=tf.reshape(data,(-1,dim_in))
        out=fc_layer(name,layer,dim_in, dim_out,wd_w,wd_b,activate,with_bn=False,init_params_flag=init_params_flag,is_train=is_train)
        return tf.reshape(out,(-1,node_num,dim_out))

def fc_layer(name,input_layer,dim_in,dim_out,wd_w=0.1,wd_b=None,activate=None,with_bn=False,init_params_flag=True,params=None,is_train=True):
    if not init_params_flag:
        tf.get_variable_scope().reuse_variables()
    w = _variable_with_weight_decay('weights', [dim_in, dim_out],
        initializer_name="normal", wd=wd_w)
    b = _variable_with_weight_decay('biases', [dim_out],
        initializer_name="zero",wd=wd_b)
    pre_activate=tf.nn.bias_add(tf.matmul(input_layer,w),b)
    # bn
    if with_bn:
        pre_activate=batch_normalization(name,pre_activate,dim_out,init_params_flag,is_train=is_train)
    #
    if activate is None:
        layer = pre_activate
    else:
        layer = activate(pre_activate)
    return layer


def graph_dropout_layer(layer,node_num,input_dim,dropout_rate):
    layer=tf.reshape(layer,(-1, node_num*input_dim))
    layer = dropout_layer(layer,dropout_rate)
    return tf.reshape(layer,(-1, node_num, input_dim))
def dropout_layer(data,dropout_rate):
    return tf.nn.dropout(data,1.0-dropout_rate)

def graph_inner_product_decoder(layer):
    layer_t=tf.transpose(layer,[0,2,1])
    adj=tf.matmul(layer,layer_t)
    return adj

def graph_distmult_decoder(name,layer,dim_in,wd_w=None,with_bn=False,init_params_flag=True,params=None):
    with tf.variable_scope(name) as scope:
        if not init_params_flag:
            tf.get_variable_scope().reuse_variables()
        w = _variable_with_weight_decay('weights', [dim_in],
            initializer_name="normal", wd=wd_w)
        layer_t=tf.transpose(layer,[0,2,1])
        adj=tf.matmul(w*layer,layer_t)
    return adj
