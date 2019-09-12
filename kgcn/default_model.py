import tensorflow as tf

class DefaultModel:
    def get_placeholders(self,info,config,batch_size,placeholder_names):
        adj_channel_num=info.adj_channel_num
        placeholders = {
            'adjs':[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)) for a in range(adj_channel_num)] for b in range(batch_size)],
            'nodes': tf.placeholder(tf.int32, shape=(batch_size,info.graph_node_num),name="node"),
            'node_label': tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.label_dim),name="node_label"),
            'mask_node_label': tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num),name="node_mask_label"),
            'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
            'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
            'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
            'is_train': tf.placeholder(tf.bool, name="is_train"),
            'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
        }

        placeholders['preference_label_list']= tf.placeholder(tf.int64, shape=(batch_size,None,6),name="preference_label_list")
        placeholders['label_list']= tf.placeholder(tf.int64, shape=(batch_size,None,2),name="label_list")
        if info.feature_enabled:
            placeholders['features']=tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.feature_dim),name="feature")
        else:
            placeholders['features']=None
        return {name:placeholders[name] for name in placeholder_names}


