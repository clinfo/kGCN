import kgcn.layers as layers
import tensorflow_federated as tff
import tensorflow as tf


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

def build_model_discriptor_gcn(max_n_atoms, max_n_types, num_classes, num_descriptors_features):
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    input_descriptors = tf.keras.Input(
        shape=(num_descriptors_features), name='descriptor')
    # for graph
    h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)    
    h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)        
    h = layers.GraphGather()(h)

    # for descriptors
    hd = tf.keras.layers.Dense(128, activation='relu')(input_descriptors)
    hd = tf.keras.layers.Dropout(0.5)(hd)    
    hd = tf.keras.layers.Dense(64, activation='relu')(hd)
    hd = tf.keras.layers.Dropout(0.5)(hd)        
    h = tf.keras.layers.Concatenate()([h, hd])
    
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
        logits = tf.keras.layers.Dense(num_classes, name="dense", activation='softmax')(h)        
    return tf.keras.Model(inputs=[input_adjs, input_features], outputs=logits)

def build_multimodel_gcn(max_n_atoms, max_n_types, num_classes, protein_max_seqlen, length_one_letter_aa, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    input_adjs = tf.keras.Input(
        shape=(1, max_n_atoms, max_n_atoms), name='adjs', sparse=False)
    input_features = tf.keras.Input(
        shape=(max_n_atoms, max_n_types), name='features')
    input_protein_seq = tf.keras.Input(shape=(protein_max_seqlen), name='protein_seq')    
    # for graph
    #h = layers.GraphConvFL(64, 1)(input_features, input_adjs)
    h = layers.GINFL(128, 1)(input_features, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    #h = layers.GraphConvFL(64, 1)(h, input_adjs)
    h = layers.GINFL(128, 1)(h, input_adjs)
    h = tf.keras.layers.ReLU()(h)
    h = layers.GraphGather()(h)

    # for protein sequence
    h_seq = tf.keras.layers.Embedding(length_one_letter_aa, 128, input_length=protein_max_seqlen)(input_protein_seq)
    stride = 4
    #h_seq = tf.keras.layers.GlobalAvePool1D()(h_seq)
    h_seq = tf.keras.layers.Dense(64, activation='relu')(h_seq)
    h_seq = tf.keras.layers.Dropout(0.5)(h_seq)
    h_seq = tf.keras.layers.GlobalAveragePooling1D()(h_seq)
    h = tf.keras.layers.Concatenate()([h, h_seq])
    h = tf.keras.layers.Dropout(0.5)(h)    
    logits = tf.keras.layers.Dense(1, name="logits", activation='sigmoid', bias_initializer=output_bias)(h)
    return tf.keras.Model(inputs=[input_adjs, input_features, input_protein_seq], outputs=logits)
