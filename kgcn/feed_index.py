import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np


def construct_feed(batch_idx,placeholders,data,batch_size,info,dropout_rate=0.0, **kwargs):
    graph_index_list=info.graph_index_list
    graph_index_list_length=max(len(l) for l in graph_index_list)
    adjs=data.adjs
    features=data.features
    nodes=data.nodes
    labels=data.labels
    mask_label=data.mask_label
    node_label=data.node_label
    mask_node_label=data.mask_node_label
    sequences=data.sequences
    sequences_len=data.sequences_len
    vector_modal=data.vector_modal
    enabled_node_nums=data.enabled_node_nums

    feed_dict={}
    if batch_size is None:
        batch_size=len(batch_idx)
    for key,pl in placeholders.items():
        if key=="adjs":
            b_shape=None
            for p,p_pl in enumerate(pl):
                for b,b_pl in enumerate(p_pl):
                    for ch,ab_pl in enumerate(b_pl):
                        if b <len(batch_idx):
                            idx=graph_index_list[batch_idx[b]][p]
                            b_shape=adjs[idx][ch][2]
                            feed_dict[ab_pl]=tf.SparseTensorValue(adjs[idx][ch][0],adjs[idx][ch][1],adjs[idx][ch][2])
                        else:
                            dummy_idx=np.zeros((0,2),dtype=np.int32)
                            dummy_val=np.zeros((0,),dtype=np.float32)
                            feed_dict[ab_pl]=tf.SparseTensorValue(dummy_idx,dummy_val,b_shape)
        elif key=="features" and features is not None:
            for p,p_pl in enumerate(pl):
                temp_features=np.zeros((batch_size,features.shape[1],features.shape[2]),dtype=np.float32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                temp_features[:len(idx),:,:]=features[idx,:,:]
                feed_dict[p_pl]=temp_features
        elif key=="nodes" and features is None:
            for p,p_pl in enumerate(pl):
                temp_nodes=np.zeros((batch_size,nodes.shape[1]),dtype=np.int32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                temp_nodes[:len(idx),:]=nodes[idx,:]
                feed_dict[p_pl]=temp_nodes
        elif key=="labels":
            for p,p_pl in enumerate(pl):
                temp_labels=np.zeros((batch_size,labels.shape[1]),dtype=np.int32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                temp_labels[:len(idx),:]=labels[idx,:]
                feed_dict[p_pl]=temp_labels
        elif key=="mask":
            for p,p_pl in enumerate(pl):
                mask=np.zeros((batch_size,),np.float32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                mask[:len(idx)]=1
                feed_dict[p_pl]=mask
        elif key=="mask_label":
            for p,p_pl in enumerate(pl):
                temp_mask_label=np.zeros((batch_size,labels.shape[1]),np.float32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                temp_mask_label[:len(idx),:]=mask_label[idx,:]
                feed_dict[p_pl]=temp_mask_label
        elif key=="node_label":
            for p,p_pl in enumerate(pl):
                temp_labels=np.zeros((batch_size,node_label.shape[1],node_label.shape[2]),dtype=np.float32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                temp_labels[:len(idx),:]=node_label[idx,:,:]
                feed_dict[pl]=temp_labels
        elif key=="mask_node_label":
            for p,p_pl in enumerate(pl):
                temp_labels=np.zeros((batch_size,mask_node_label.shape[1]),dtype=np.float32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                temp_labels[:len(idx),:]=mask_node_label[idx,:]
                feed_dict[pl]=temp_labels
        elif key=="mask_node" and enabled_node_nums is not None:
            temp_mask_label=np.zeros((batch_size,info.graph_node_num),np.float32)
            lengths=enabled_node_nums[batch_idx]
            for j,l in enumerate(lengths):
                temp_mask_label[j,:l]=1.0
            feed_dict[pl]=temp_mask_label
        elif key=="sequences" and sequences is not None:
            for p,p_pl in enumerate(pl):
                seqs=np.zeros((batch_size,sequences.shape[1]),np.int32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                seqs[:len(idx),:]=sequences[idx,:]
                feed_dict[pl]=seqs
        elif key=="sequences_len" and sequences_len is not None:
            for p,p_pl in enumerate(pl):
                seqs_len=np.zeros((batch_size,2),np.int32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                seqs_len[:len(idx),1]=sequences_len[idx]-1
                seqs_len[:len(idx),0]=range(len(idx))
                feed_dict[pl]=seqs_len
        elif key=="dropout_rate":
            feed_dict[pl]=dropout_rate
        elif key=="enabled_node_nums" and enabled_node_nums is not None:
            temp_enabled_node_nums=np.zeros((batch_size, ), np.int32)
            temp_enabled_node_nums[:len(batch_idx)]=np.squeeze(enabled_node_nums[batch_idx])
            feed_dict[pl]=temp_enabled_node_nums
        elif key=="epsilon":
            eps=np.random.standard_normal(pl.shape)
            feed_dict[pl]=eps

    return feed_dict


