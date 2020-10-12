from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense

enabled_batched = False
enabled_bspmm = False
enabled_bconv = False


def load_bspmm(args):
    global enabled_batched
    global enabled_bspmm
    global enabled_bconv
    external_path = "./"
    if args.batched and os.path.exists(external_path+"batched.so"):
        enabled_batched = True
    elif args.bspmm and os.path.exists(external_path+"bspmm.so"):
        enabled_bspmm = True
    elif args.bconv and os.path.exists(external_path+"bconv.so"):
        enabled_bconv = True


class GraphConv(Layer):
    def __init__(self, output_dim, adj_channel_num, initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.adj_channel_num = adj_channel_num
        self.initializer = initializer
        if enabled_bspmm:
            import kgcn.bspmm_call as bspmm
            self.bspmm_obj = bspmm.BatchedSpMM()
        if enabled_bconv:
            import kgcn.bconv_call as bconv
            self.bconv_obj = bconv.BatchedConv()
        if enabled_batched:
            import kgcn.batched_call as batched
            self.bspmdt_obj = batched.BatchedSpMDT()
        super(GraphConv, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        adj_channel_num = self.adj_channel_num
        # Create a trainable weight variable for this layer.
        self.w = []
        self.bias = []
        for i in range(adj_channel_num):
            self.w.append(self.add_weight(name='kernel'+str(i),
                                          shape=(int(input_shape[2]), int(self.output_dim)),
                                          initializer=self.initializer,
                                          trainable=True))
            self.bias.append(self.add_weight(name='bias'+str(i),
                                             shape=(1, int(self.output_dim)),
                                             initializer='zeros',
                                             trainable=True))
        super(GraphConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, adj=None):
        adjs = adj
        adj_channel_num = self.adj_channel_num
        batch_size = inputs.shape[0]
        if enabled_bconv:
            print("## bconv ##")
            # o = []  # unused variable
            # Batched Convolution kernel
            fw = [[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
            for batch_idx in range(batch_size):
                for adj_ch in range(adj_channel_num):
                    fb = inputs[batch_idx, :, :]
                    fw[batch_idx][adj_ch] = tf.matmul(fb, self.w[adj_ch])+self.bias[adj_ch]
            oo = self.bconv_obj.call(adjs, fw)
            return tf.stack(oo)
        elif enabled_bspmm:
            print("## bspmm ##")
            o = []
            for adj_ch in range(adj_channel_num):
                adj_list = [adjs[i][adj_ch] for i in range(batch_size)]
                fw_list = [None for _ in range(batch_size)]
                for batch_idx in range(batch_size):
                    fb = inputs[batch_idx, :, :]
                    fw_list[batch_idx] = tf.matmul(fb, self.w[adj_ch]) + self.bias[adj_ch]
                oo = self.bspmm_obj.call(adj_list, fw_list)
                o = oo if not o else tf.add(o, oo)
            return tf.stack(o)
        elif enabled_batched:
            print("## batched ##")
            # Batched version
            o = [None for _ in range(adj_channel_num)]
            input_row = inputs.shape[1]
            input_col = inputs.shape[2]
            for adj_ch in range(adj_channel_num):
                # w_col = tf.shape(self.w[adj_ch])[1]  # unused variable
                fs = tf.reshape(inputs, [batch_size*input_row, input_col])
                fw = tf.matmul(fs, self.w[adj_ch])+self.bias[adj_ch]
                adj_list = [adjs[batch_idx][adj_ch] for batch_idx in range(batch_size)]
                o[adj_ch] = self.bspmdt_obj.call(adj_list, fw)
            o = tf.reduce_sum(o, 0)
            return tf.stack(o)
        else:
            # graph conv. without bspmm
            o = [[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
            for batch_idx in range(batch_size):
                for adj_ch in range(adj_channel_num):
                    adj = adjs[batch_idx][adj_ch]
                    fb = inputs[batch_idx, :, :]
                    fw = tf.matmul(fb, self.w[adj_ch])+self.bias[adj_ch]
                    #el = tf.sparse_tensor_dense_matmul(adj, fw)
                    el = tf.sparse.sparse_dense_matmul(adj, fw)                    
                    #
                    o[batch_idx][adj_ch] = el
                o[batch_idx] = tf.add_n(o[batch_idx])
        return tf.stack(o)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


class GraphMaxPooling(Layer):
    def __init__(self, adj_channel_num, **kwargs):
        self.adj_channel_num = adj_channel_num
        super(GraphMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        super(GraphMaxPooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, adj=None):
        adj_channel_num = self.adj_channel_num
        batch_size = inputs.shape[0]
        dim = inputs.shape[2]
        o = [[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            for adj_ch in range(adj_channel_num):
                vec = [None for _ in range(dim)]
                for k in range(dim):
                    adj_mat = adj[batch_idx][adj_ch]
                    fb = inputs[batch_idx, :, k]
                    x = adj_mat*fb
                    d = tf.sparse_tensor_to_dense(x)
                    el = tf.reduce_max(d, axis=1)
                    # el=tf.sparse_reduce_max(x,axis=1)
                    vec[k] = el
                o[batch_idx][adj_ch] = tf.stack(vec, axis=1)
            o[batch_idx] = tf.add_n(o[batch_idx])
        output = tf.stack(o)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class GraphGather(Layer):
    def __init__(self, **kwargs):
        super(GraphGather, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        super(GraphGather, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class GraphBatchNormalization(Layer):
    def __init__(self, bn_name=None, **kwargs):
        self.bn_name = bn_name
        super(GraphBatchNormalization, self).__init__(**kwargs)

    def normalization_shape(self, input_shape):
        batch_size = input_shape[0]
        node_num = input_shape[1]
        input_dim = input_shape[2]
        normalize_axis_dim = -1
        if node_num is not None and input_dim is not None:
            normalize_axis_dim = node_num*input_dim
        return batch_size, normalize_axis_dim

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        self.data_shape = input_shape
        super(GraphBatchNormalization, self).build(input_shape)

    def call(self, inputs, enabled_node_nums=None, shape=None, max_node_num=None, training=True,):
        # shape needs to be explicitly specified. tf cannot automatically keep track of the size.
        # initialize
        if shape is None:
            batch_size = self.data_shape[0]
            node_num = max_node_num
            input_dim = self.data_shape[2]
        else:
            batch_size = shape[0]
            node_num = shape[1]
            input_dim = shape[2]
        # computing
        if enabled_node_nums is not None:
            inputs.set_shape([batch_size, node_num, input_dim])
            extracted_nodes = [feature_map[:enabled_node_num] for feature_map, enabled_node_num in
                               zip(tf.unstack(inputs), tf.unstack(enabled_node_nums))]
            stacked_nodes = tf.concat(extracted_nodes, 0)
            normalized_data = tf.keras.layers.BatchNormalization(trainable=training, name=self.bn_name)(stacked_nodes)
            split_data = tf.split(normalized_data, enabled_node_nums)
            padded_data = [tf.pad(feature_map, [[0, node_num-tf.shape(feature_map)[0]], [0, 0]])
                           for feature_map in split_data]
            output = tf.stack(padded_data)
            output.set_shape([batch_size, node_num, input_dim])
            return output
        else:
            # if node_num is not None and input_dim is not None:
            layer = tf.reshape(inputs, (-1, input_dim))
            layer = tf.keras.layers.BatchNormalization(trainable=training, name=self.bn_name)(layer)
            layer = tf.reshape(layer, (batch_size, -1, input_dim))
            return layer

    def compute_output_shape(self, input_shape):
        return input_shape


class GraphDense(Dense):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GraphDense, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        self.data_shape = input_shape
        super(GraphDense, self).build(input_shape)

    def call(self, inputs, enabled_node_nums=None, shape=None, max_node_num=None, **kwargs):
        # initialize
        # Dynamic
        if shape is None:
            batch_size = self.data_shape[0]
            node_num = max_node_num
            input_dim = self.data_shape[2]
        else:
            batch_size = shape[0]
            node_num = shape[1]
            input_dim = shape[2]
        if enabled_node_nums is not None:
            inputs.set_shape([batch_size, node_num, input_dim])
            extracted_nodes = [feature_map[:enabled_node_num] for feature_map, enabled_node_num in
                               zip(tf.unstack(inputs), tf.unstack(enabled_node_nums))]
            stacked_nodes = tf.concat(extracted_nodes, 0)
            out = super(GraphDense, self).call(stacked_nodes, **kwargs)
            split_data = tf.split(out, enabled_node_nums)
            padded_data = [tf.pad(feature_map, [[0, node_num-tf.shape(feature_map)[0]], [0, 0]])
                           for feature_map in split_data]
            output = tf.stack(padded_data)
            output.set_shape([batch_size, node_num, self.output_dim])
            return output
        else:
            dims = inputs.shape
            batch_size, node_num, dim_in = dims[0], dims[1], dims[2]
            # computing
            data = tf.reshape(inputs, (-1, dim_in))
            out = super(GraphDense, self).call(data, **kwargs)
            out = tf.reshape(out, (batch_size, -1, self.output_dim))
            return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


class GraphDecoderInnerProd(Layer):
    def __init__(self, **kwargs):
        super(GraphDecoderInnerProd, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        super(GraphDecoderInnerProd, self).build(input_shape)

    def call(self, inputs, **kwargs):
        layer = inputs
        layer_t = tf.transpose(layer, [0, 2, 1])
        adj = tf.matmul(layer, layer_t)
        return adj

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1]


class GraphDecoderDistMult(Layer):
    def __init__(self, initializer='glorot_uniform', **kwargs):
        self.initializer = initializer
        super(GraphDecoderDistMult, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        self.w = []
        self.w.append(self.add_weight(name='kernel',
                                      shape=(int(input_shape[2]),),
                                      initializer=self.initializer,
                                      trainable=True))
        super(GraphDecoderDistMult, self).build(input_shape)

    def call(self, inputs, **kwargs):
        layer = inputs
        layer_t = tf.transpose(layer, [0, 2, 1])
        adj = tf.matmul(self.w[0]*layer, layer_t)
        return adj

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1]

class DistMult(Layer):
    def __init__(self, initializer='glorot_uniform', adj_channel_num=1, **kwargs):
        self.initializer = initializer
        self.adj_channel_num = adj_channel_num
        super(DistMult, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        self.w = []
        self.w.append(self.add_weight(name='kernel',
                                      shape=(self.adj_channel_num,int(input_shape[2]),),
                                      initializer=self.initializer,
                                      trainable=True))
        super(DistMult, self).build(input_shape)
    # batch_size x dim
    def compute_score(self, layer1, layer2, channel,**kwargs):
        ww=self.w[0]
        ww_channel=tf.gather(ww,channel,axis=0)
        score=tf.reduce_sum(layer1*layer2*ww_channel,axis=1)
        return score
    
    def compute_left_prediction(self, layer, right_layer, channel,**kwargs):
        ww=self.w[0]
        ww_channel=tf.gather(ww,channel,axis=0)
        #ww_channel/right_layer: batch x dim
        #layer: node_num x dim
        layer_a=right_layer*ww_channel
        layer_b=tf.transpose(layer,[1,0])
        score=tf.matmul(layer_a,layer_b)
        #score: batch x node_num
        return score
 
    def compute_right_prediction(self, left_layer, layer, channel,**kwargs):
        ww=self.w[0]
        ww_channel=tf.gather(ww,channel,axis=0)
        #ww_channel/right_layer: batch x dim
        #layer: batch x node_num x dim
        temp=tf.expand_dims(left_layer*ww_channel,2)
        score=tf.matmul(layer,temp)
        #score: batch x node_num
        score=tf.squeeze(score,[2])
        return score

    def call(self, inputs, **kwargs):
        layer = inputs
        layer_t = tf.transpose(layer, [0, 2, 1])
        adjs=[]
        for i in range(self.adj_channel_num):
            adj = tf.matmul(self.w[0][i]*layer, layer_t)
            adjs.append(adj)
        return tf.transpose(tf.stack(adjs),[1,0,2,3])
    def compute_output_shape(self, input_shape):
        return input_shape[0], adj_channel_num, input_shape[1], input_shape[1]




class BatchGraphConv(Layer):
    def __init__(self, output_dim, adj_channel_num=1, initializer='glorot_uniform', input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.adj_channel_num = adj_channel_num
        self.initializer = initializer
        self.input_dim = input_dim
        super(BatchGraphConv, self).__init__(**kwargs)

    def build(self, input_shape):
        adj_channel_num = self.adj_channel_num
        # Create a trainable weight variable for this layer.
        # self.ww
        # self.bias=[]
        for i in range(adj_channel_num):
            input_dim = input_shape[0][1] if self.input_dim is None else self.input_dim
            self.w = self.add_weight(name='kernel'+str(i),
                                     shape=(int(input_dim), int(self.output_dim)),
                                     initializer=self.initializer,
                                     trainable=True)
            self.bias = self.add_weight(name='bias'+str(i),
                                        shape=(int(self.output_dim)),
                                        initializer='zeros',
                                        trainable=True)
        super(BatchGraphConv, self).build(input_shape)  # Be sure to call this somewhere!ef __init__(self, out_dim, **kwargs):

    def call(self, inputs, **kwargs):
        net = inputs[0]
        adj = inputs[1]
        net = tf.matmul(net, self.w)
        net = tf.nn.bias_add(net, self.bias)
        net = tf.sparse_tensor_dense_matmul(adj, net)
        net = tf.nn.relu(net)
        return net

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.output_dim

class GINAggregate(Layer):
    def __init__(self, adj_channel_num, initializer='zeros', **kwargs):
        self.adj_channel_num = adj_channel_num
        self.initializer = initializer
        if enabled_bspmm:
            import kgcn.bspmm_call as bspmm
            self.bspmm_obj = bspmm.BatchedSpMM()
        if enabled_bconv:
            import kgcn.bconv_call as bconv
            self.bconv_obj = bconv.BatchedConv()
        if enabled_batched:
            import kgcn.batched_call as batched
            self.bspmdt_obj = batched.BatchedSpMDT()
        super(GINAggregate, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        adj_channel_num = self.adj_channel_num
        self.epsilon = []
        for i in range(adj_channel_num):
            self.epsilon.append(self.add_weight(name='epsilon'+str(i),
                                          shape=(),
                                          initializer=self.initializer,
                                          trainable=True))
        super(GINAggregate, self).build(input_shape)

    def call(self, inputs, adj=None):
        adjs = adj
        adj_channel_num = self.adj_channel_num
        batch_size = inputs.shape[0]
        if enabled_bconv:
            print("## bconv ##")
            fw = [[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
            for batch_idx in range(batch_size):
                for adj_ch in range(adj_channel_num):
                    fw[batch_idx][adj_ch] = inputs[batch_idx, :, :]
            oo = self.bconv_obj.call(adjs, fw)
            return tf.stack(oo)
        elif enabled_bspmm:
            print("## bspmm ##")
            o = []
            for adj_ch in range(adj_channel_num):
                adj_list = [adjs[i][adj_ch] for i in range(batch_size)]
                fw_list = [None for _ in range(batch_size)]
                for batch_idx in range(batch_size):
                    fw_list[batch_idx] = inputs[batch_idx, :, :]
                oo = self.bspmm_obj.call(adj_list, fw_list)
                o = oo if not o else tf.add(o, oo)
            return tf.stack(o)
        elif enabled_batched:
            print("## batched ##")
            # Batched version
            o = [None for _ in range(adj_channel_num)]
            input_row = inputs.shape[1]
            input_col = inputs.shape[2]
            for adj_ch in range(adj_channel_num):
                # w_col = tf.shape(self.w[adj_ch])[1]  # unused variable
                fs = tf.reshape(inputs, [batch_size*input_row, input_col])
                adj_list = [adjs[batch_idx][adj_ch] for batch_idx in range(batch_size)]
                o[adj_ch] = self.bspmdt_obj.call(adj_list, fs)
            o = tf.reduce_sum(o, 0)
            return tf.stack(o)
        else:
            # graph conv. without bspmm
            o = [[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
            for batch_idx in range(batch_size):
                for adj_ch in range(adj_channel_num):
                    adj = adjs[batch_idx][adj_ch]
                    fw = inputs[batch_idx, :, :]
                    el = tf.sparse_tensor_dense_matmul(adj, fw)
                    o[batch_idx][adj_ch] = self.epsilon[adj_ch]*fw+el
                o[batch_idx] = tf.add_n(o[batch_idx])
            output = tf.stack(o)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class GAT(Layer):
    """Graph attentional layer: https://arxiv.org/abs/1710.10903
        Note: this layer does not include weight parameter W
              for the flexible design of neural networks.
        For example , the following pseudo code is an implementation of 
        the graph attention network in the thesis,
        ```python
        h=GraphDense(input)
        out=GAT(h)
        ```

    """
    def __init__(self, adj_channel_num, initializer='glorot_uniform',input_dim=None, **kwargs):
        self.adj_channel_num = adj_channel_num
        self.initializer = initializer
        self.input_dim = input_dim
        super(GAT, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        adj_channel_num = self.adj_channel_num
        self.weight_a = []
        self.w = []
        for i in range(adj_channel_num):
            input_dim = input_shape[2] if self.input_dim is None else self.input_dim
            weight_a = self.add_weight(name='weight_a'+str(i),
                                     shape=(int(input_dim)*2,1),
                                     initializer=self.initializer,
                                     trainable=True)
            self.weight_a.append(weight_a)
        super(GAT, self).build(input_shape)

    def call(self, inputs, adj=None):
        adjs = adj
        adj_channel_num = self.adj_channel_num
        batch_size = inputs.shape[0]
        max_node_num = inputs.shape[1]
        # graph conv. without bspmm
        o = [[None for _ in range(adj_channel_num)] for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            for adj_ch in range(adj_channel_num):
                adj = adjs[batch_idx][adj_ch]
                fw = inputs[batch_idx, :, :]
                x=fw
                idx=adj.indices
                a1=tf.gather(x,idx[:,1])
                a2=tf.gather(x,idx[:,0])
                # ii: node_num x #edges
                ii=tf.transpose(tf.one_hot(idx[:,0],max_node_num))
                ##
                aa=tf.concat([a1,a2],axis=1)
                layer=tf.matmul(aa,self.weight_a[adj_ch])
                layer=tf.nn.leaky_relu(layer)
                e=tf.exp(layer)
                denom=tf.matmul(ii,e)
                denom_e=tf.gather(denom,idx[:,1])
                alpha=e/(denom_e+1.0e-10)
                ##
                r=tf.matmul(ii,alpha*a1)
                o[batch_idx][adj_ch] = tf.nn.sigmoid(r)
            o[batch_idx] = tf.add_n(o[batch_idx])
        output = tf.stack(o)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


