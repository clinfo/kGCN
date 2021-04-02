import kgcn.layers as layers
import tensorflow_federated as tff
import tensorflow as tf
from tensorflow.keras import layers as klayers
import scipy.sparse as sp
import spektral.layers as slayers


def build_model_gin(max_n_atoms, max_n_types, num_classes):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    h = layers.GINFL(128, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GINFL(128, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    if num_classes == 1:
        h = tf.keras.layers.Dense(128, name="dense", activation='relu')(h)        
        logits = tf.keras.layers.Dense(num_classes, name="out")(h)
    else:
        logits = tf.keras.layers.Dense(num_classes, name="dense", activation='softmax')(h)        
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)

def build_model_gcn(max_n_atoms, max_n_types, num_classes):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    # for graph
    h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)
    if num_classes == 1:
        logits = tf.keras.layers.Dense(num_classes, name="dense")(h)
    else:
        logits = tf.keras.layers.Dense(num_classes, name="dense", activation='softmax')(h)        
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)

def build_model_discriptor_gcn(max_n_atoms, max_n_types, num_classes,
                               num_descriptors_features, initializer='glorot_uniform'):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    input_descriptors = tf.keras.Input(
        shape=(num_descriptors_features), name='descriptor')
    # for graph
    h = layers.GraphConvFL(128, 1, initializer)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    #h = tf.keras.layers.Dropout(0.5)(h)
    h = tf.keras.layers.BatchNormalization()(h)        
    h = layers.GraphConvFL(128, 1, initializer)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)    
    #h = tf.keras.layers.Dropout(0.5)(h)        
    h = layers.GraphGather()(h)
    # for descriptors
    hd = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(input_descriptors)
    hd = tf.keras.layers.BatchNormalization()(hd)
    #hd = tf.keras.layers.Dropout(0.5)(hd)
    hd = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(hd)
    hd = tf.keras.layers.BatchNormalization()(hd)    
    h = tf.keras.layers.Concatenate()([h, hd])
    #h = hd
    if num_classes == 1:
        logits = tf.keras.layers.Dense(num_classes)(h)
    else:
        logits = tf.keras.layers.Dense(num_classes, activation='softmax')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_descriptors], outputs=logits)

def build_model_gat(max_n_atoms, max_n_types, num_classes, num_heads=3):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    # for graph
    h = layers.GATFL(32, num_heads, 1)(input_features, input_adjs)
    h = layers.GATFL(32, num_heads, 1)(h, input_adjs)
    h = layers.GraphGather()(h)
    if num_classes == 1:
        logits = tf.keras.layers.Dense(num_classes, name="dense")(h)
    else:
        logits = tf.keras.layers.Dense(num_classes, name="dense")(h)        
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)

def build_multimodel_gcn(max_n_atoms, max_n_types, num_classes, protein_max_seqlen, length_one_letter_aa, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    input_adjs = tf.keras.Input(
        shape=(max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, 23), name='features')
    input_protein_seq = tf.keras.Input(shape=(protein_max_seqlen), name='protein_seq')    
    # for graph
    #h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = slayers.GCSConv(128)([input_features, input_adjs])
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)        
    #h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = slayers.GCSConv(64)([h, input_adjs])
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = slayers.GCSConv(32)([h, input_adjs])
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)            
    h = layers.GraphGather()(h)
    # for protein sequence
    h_seq = tf.keras.layers.Embedding(length_one_letter_aa, 64, input_length=protein_max_seqlen)(input_protein_seq)
    h_seq = tf.keras.layers.Conv1D(32, 4, 3)(h_seq)
    h_seq = tf.keras.layers.AveragePooling1D(3)(h_seq)
    h_seq = tf.keras.layers.Conv1D(16, 4, 3)(h_seq)
    h_seq = tf.keras.layers.AveragePooling1D(2)(h_seq)
    h_seq = tf.keras.layers.Dense(32, activation='relu')(h_seq)    
    h_seq = tf.keras.layers.Flatten()(h_seq)
    print(h_seq.shape)
    #h_seq = tf.keras.layers.GlobalAvePool1D()(h_seq)
    #h_seq = tf.keras.layers.GlobalAveragePooling1D()(h_seq)
    h = tf.keras.layers.Concatenate()([h, h_seq])
    h = tf.keras.layers.Dropout(0.5)(h)    
    logits = tf.keras.layers.Dense(1, name="logits", activation='sigmoid', bias_initializer=output_bias)(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_protein_seq], outputs=logits)


def build_dgl_model_discriptor_gcn(max_n_atoms, max_n_types, num_classes,
                                   num_descriptors_features, initializer='glorot_uniform'):
    input_adjs = tf.keras.Input(
        shape=(max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    input_descriptors = tf.keras.Input(
        shape=(num_descriptors_features), name='descriptor')
    # for graph
    layer_name = 'GCSConv'
    _gcn_layers = ['APPNPConv', 'ARMAConv', 'ChebConv',
                   'DiffusionConv', 'GATConv', 'GCNConv', 'GCSConv']
    
    h = getattr(slayers, layer_name)(128, activation='relu')([input_features, input_adjs])
    h = tf.keras.layers.BatchNormalization()(h)    
    h = getattr(slayers, layer_name)(64, activation='relu')([h, input_adjs])
    h = tf.keras.layers.BatchNormalization()(h)
    h = getattr(slayers, layer_name)(32, activation='relu')([h, input_adjs])
    h = slayers.GlobalAttnSumPool()(h)    
    # for descriptors

    original_shape = input_descriptors.shape
    # print(original_shape)

    inputs = input_descriptors
    #inputs = tf.keras.layers.Dropout(0.2)(inputs)    
    hd = tf.expand_dims(inputs, axis=1)
    hd = tf.keras.layers.LocallyConnected1D(num_descriptors_features, 1, activation='relu')(hd)
    hd = tf.keras.layers.Reshape([num_descriptors_features,])(hd)    
    hdl = tf.keras.layers.Dense(num_descriptors_features, activation='relu', kernel_initializer=initializer)(inputs)
    hdl = tf.keras.layers.BatchNormalization()(hdl)
    hd = hd + inputs + hdl

    hd = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(hd)

    hd = inputs
    hd = tf.expand_dims(inputs, axis=1)
    hd = tf.keras.layers.LocallyConnected1D(num_descriptors_features, 1, activation='relu')(hd)
    hd = tf.keras.layers.Reshape([num_descriptors_features,])(hd)    
    hdl = tf.keras.layers.Dense(num_descriptors_features, activation='relu', kernel_initializer=initializer)(inputs)
    hdl = tf.keras.layers.BatchNormalization()(hdl)
    hd = hd + inputs + hdl
    
    #hd = inputs
    #hd = hdl
    #hd = input_descriptors    
    #hd = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(input_descriptors)
    # hd = tf.keras.layers.Dense(num_descriptors_features, activation='relu', kernel_initializer=initializer)(hd)
    # hd = tf.keras.layers.Dropout(0.5)(hd)

    hd = tf.keras.layers.BatchNormalization()(hd)
    hd = tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer)(hd)
    # hd = tf.keras.layers.Dense(7, activation='tanh', kernel_initializer=initializer)(hd)
    hd = tf.keras.layers.BatchNormalization()(hd)    
    h = tf.keras.layers.Concatenate()([h, hd])
    #h = hd
    if num_classes == 1:
        logits = tf.keras.layers.Dense(num_classes)(h)
    else:
        logits = tf.keras.layers.Dense(num_classes, activation='softmax')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_descriptors], outputs=logits)

def build_optuna_model(gcn_hiddens, gcn_layers, linear_hiddens,
                       max_n_atoms, max_n_types, num_classes,
                       num_descriptors_features, initializer='glorot_uniform'):
    input_adjs = tf.keras.Input(
        shape=(max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    input_descriptors = tf.keras.Input(
        shape=(num_descriptors_features), name='descriptor')
    # gcn
    h = input_features
    for g_h, g_layer in zip(gcn_hiddens, gcn_layers):
        h = getattr(slayers, g_layer)(g_h, activation='relu')([h, input_adjs])
        h = tf.keras.layers.BatchNormalization()(h)        
    h = slayers.GlobalAttnSumPool()(h)

    # descriptors
    lh = input_descriptors
    lh = tf.keras.layers.Dense(128, activation='relu')(lh)
    lh = tf.keras.layers.BatchNormalization()(lh)    
    for l_h in linear_hiddens:
        lh = tf.keras.layers.Dense(l_h, activation='relu')(lh)
        lh = tf.keras.layers.BatchNormalization()(lh)
        
    h = tf.keras.layers.Concatenate()([h, lh])
    if num_classes == 1:
        logits = tf.keras.layers.Dense(num_classes)(h)
    else:
        logits = tf.keras.layers.Dense(num_classes, activation='softmax')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_descriptors], outputs=logits)


def build_optuna_model(gcn_hiddens, gcn_layers, linear_hiddens,
                       max_n_atoms, max_n_types, num_classes,
                       num_descriptors_features, initializer='glorot_uniform'):
    input_adjs = tf.keras.Input(
        shape=(max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    input_descriptors = tf.keras.Input(
        shape=(num_descriptors_features), name='descriptor')
        
    # gcn
    h = input_features
    for g_h, g_layer in zip(gcn_hiddens, gcn_layers):
        h = getattr(slayers, g_layer)(g_h, activation='relu')([h, input_adjs])
        h = tf.keras.layers.BatchNormalization()(h)        
    h = slayers.GlobalAttnSumPool()(h)

    # descriptors
    lh = input_descriptors
    lh = tf.keras.layers.Dense(128, activation='relu')(lh)
    lh = tf.keras.layers.BatchNormalization()(lh)    
    for l_h in linear_hiddens:
        lh = tf.keras.layers.Dense(l_h, activation='relu')(lh)
        lh = tf.keras.layers.BatchNormalization()(lh)
        
    h = tf.keras.layers.Concatenate()([h, lh])
    if num_classes == 1:
        logits = tf.keras.layers.Dense(num_classes)(h)
    else:
        logits = tf.keras.layers.Dense(num_classes, activation='softmax')(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_descriptors], outputs=logits)


def build_multimodal_optuna_model(gcn_hiddens, gcn_layers, linear_hiddens,
                                  max_n_atoms, max_n_types, num_classes,
                                  num_descriptors_features, initializer='glorot_uniform'):
    pass
