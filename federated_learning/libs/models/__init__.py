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
