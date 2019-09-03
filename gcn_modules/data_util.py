import tensorflow as tf
import numpy as np
import joblib
import time
import json
import argparse
import importlib
import os

## gcn project
#import model
#import layers
#import visualization

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def align_size(adjs,maxN):
    flag = (type(adjs[0][0])==tuple)
    for i in range(len(adjs)):
        for ch in range(len(adjs[0])):
            if flag:
                adjs[i][ch]=list(adjs[i][ch])
            adjs[i][ch][2]=[maxN,maxN]
    return
def dense_to_sparse(dense):
    from scipy.sparse import coo_matrix
    coo=coo_matrix(dense)
    sh=coo.shape
    val=coo.data
    sp=list(zip(coo.row,coo.col))
    return (np.array(sp),np.array(val,dtype=np.float32),np.array(sh))

def check_adj(adj):
    try:
        if len(adj)==3 and len(adj[2])==2:
            if type(adj[2][0]) not in (np.ndarray,list):
                return True
    except:
        return False
    return False

def high_order_adj(adj,order):
    from scipy.sparse import coo_matrix
    if order<=1:
        return adj
    A = coo_matrix((adj[1],np.transpose(adj[0])),shape=adj[2])
    A = A.tocsr()
    B=A
    for _ in range(order-1):
        B=B.dot(A)
    # csr -> coo -> adj
    coo=B.tocoo()
    sh=coo.shape
    val=coo.data
    val=np.ones(val.shape,np.float32)
    sp=list(zip(coo.row,coo.col))
    sp=sorted(sp)
    return (np.array(sp,np.int32),np.array(val,dtype=np.float32),np.array(sh,np.int64))

def split_adj(adjs,min_deg=1,max_deg=5):
    split_ch_num=(max_deg-min_deg+1)+1
    self_ch=max_deg-min_deg+1 # self loop
    for gid,adj_set in enumerate(adjs):
        new_adjs_all=[]
        for ch,adj in enumerate(adj_set):
            # type conversion
            adj=list(adj)
            adj[1]=adj[1].astype(np.float32)

            # computing degree
            deg={i:0 for i in range(adj[2][0])}
            for e in adj[0]:
                i=e[0]
                deg[i]+=1
            split_ch={}
            for k,v in deg.items():
                split_ch[k]=v-min_deg
                if v>max_deg:
                    split_ch[k]=max_deg-min_deg

            # splitting degree
            # set dummy to avoid zero-element matrix (It causes an error in the feed step)
            new_adjs=[[ [[0,0]] ,[0.0],adj[2]] for _ in range(split_ch_num)]
            for i,e in enumerate(adj[0]):
                v=adj[1][i]
                if e[0]==e[1]:# self loop
                    new_adjs[self_ch][0].append(e)
                    new_adjs[self_ch][1].append(v)
                else:
                    k=e[0]
                    new_adjs[split_ch[k]][0].append(e)
                    new_adjs[split_ch[k]][1].append(v)
            for m in new_adjs:
                # duplicated elements at [0, 0] causes an error in gcn.
                # if m[0] = [[0, 0], [0, 0], ...] and m[1] = [0, 1, ...], then
                # the first elements of m[0] and m[1] should be removed.
                if len(m[0]) > 2 and all(m[0][1] == [0, 0]):
                    m[0]=m[0][1:]
                    m[1]=m[1][1:]
                m[0]=np.array(m[0],np.int32)
                m[1]=np.array(m[1],np.float32)

            new_adjs_all.extend(new_adjs)
        adjs[gid]=new_adjs_all
    return adjs

def normalize_adj(adjs):
    from scipy.sparse import coo_matrix
    normalized_adjs = []
    for gid,adj_set in enumerate(adjs):
        normalized_adj_set = []
        for ch,adj in enumerate(adj_set):
            # normalize. Refer to Kipf 2017.
            adj[1][adj[1]>0]=1
            A_tilde = coo_matrix((adj[1], (adj[0][:,0],adj[0][:,1])),shape=adj[2])
            degrees=np.squeeze(np.asarray(np.sum(A_tilde, 0)))
            degrees[degrees==0] = 1
            A_hat = A_tilde / np.sqrt(np.expand_dims(degrees, 1)) / np.sqrt(degrees)
            normalized_adj = dense_to_sparse(A_hat)
            normalized_adj_set.append(normalized_adj)

        normalized_adjs.append(normalized_adj_set)
    return normalized_adjs


class DataLoadError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

def shuffle_data(data):
    idx=list(range(data.num))
    np.random.shuffle(idx)
    # convert
    if data.adjs is not None: data.adjs=np.array(data.adjs)
    # shuffle
    if data.features is not None: data.features=data.features[idx]
    if data.nodes is not None: data.nodes=data.nodes[idx]
    if data.adjs is not None: data.adjs=data.adjs[idx]
    if data.labels is not None: data.labels=data.labels[idx]
    if data.mask_label is not None: data.mask_label=data.mask_label[idx]
    if data.node_label is not None: data.node_label=data.node_label[idx]
    if data.mask_node_label is not None: data.mask_node_label=data.mask_node_label[idx]
    if data.label_list is not None: data.label_list=data.label_list[idx]
    if data.sequences is not None: data.sequences=data.sequences[idx]
    if data.sequences_len is not None: data.sequences_len=data.sequences_len[idx]
    if data.vector_modal is not None:
        data.vector_modal=[data.vector_modal[j][idx] for j in range(len(data.vector_modal))]
    if data.enabled_node_nums is not None: data.enabled_node_nums=data.enabled_node_nums[idx]

    return data


def load_and_split_data(config,filename="data.jbl",valid_data_rate=0.2):
    all_data,info=load_data(config,filename)
    train_data,valid_data=split_data(all_data,valid_data_rate)

    return all_data,train_data,valid_data,info

def load_data(config,filename="data.jbl",prohibit_shuffle=False):
    print("[LOAD]",filename)
    data=joblib.load(filename)
    # data
    ## Num x N x F
    features=None
    if "feature" in data:
        if config["with_feature"]:
            features=data["feature"]
    if features is not None and len(features)==0:
        features=None

    nodes=None
    if "node" in data:
        if config["with_node_embedding"]:
            nodes=np.array(data["node"],np.int32)
    if nodes is not None and len(nodes)==0:
        nodes=None
    ## Num x (N x N)
    normalize_flag=config["normalize_adj_flag"]
    split_flag=config["split_adj_flag"]
    order=1
    if "order" in config:
        order=config["order"]
    adj_channel_num=1
    enabled_node_nums=None
    try:
        if "multi_dense_adj" not in data:
            if "adj" in data:
                adjs=data["adj"]
            elif "dense_adj" in data:
                adjs=[dense_to_sparse(mat) for mat in data["dense_adj"]]
            else:
                print("[WARNING] adj or dense_adj are required for GCN")
                raise DataLoadError("")
            maxN=data["max_node_num"]
            if check_adj(adjs[0]):
                adjs=[[high_order_adj(adj,o) for o in range(1,order+1)] for adj in adjs]
            enabled_node_nums=[adj[0][2][0] for adj in adjs]
            #maxN=np.max([adj[2][0] for adj in adjs])
            align_size(adjs,maxN)

        else:
            enabled_node_nums=[max([len(mat) for mat in list_mat]) for list_mat in data["multi_dense_adj"]]
            adjs=[[dense_to_sparse(mat) for mat in list_mat] for list_mat in data["multi_dense_adj"]]
        if split_flag:
            adjs=split_adj(adjs)
        if normalize_flag:
            adjs=normalize_adj(adjs)
        adj_channel_num=len(adjs[0])
        enabled_node_nums=np.array(enabled_node_nums,dtype=np.int32)
    except DataLoadError:
        print("[INFO] no graph")
        adjs=None
        pass
    ## Setting label info.
    labels=None
    if "label" in data:
        labels=data["label"]
    mask_label=None
    if "mask_label" in data:
        mask_label=data["mask_label"]
    if "label_sparse" in data:
        labels=np.array(data["label_sparse"].todense())
    if "mask_label_sparse" in data:
        mask_label=np.array(data["mask_label_sparse"].todense())
    node_label=None
    if "node_label" in data:
        node_label=data["node_label"]
    mask_node_label=None
    if "mask_node_label" in data:
        mask_node_label=data["mask_node_label"]
    label_list=None
    if "label_list" in data:
        label_list=data["label_list"]
    ## Setting sequence data (multimodal)
    sequences=None
    sequences_len=None
    if "sequence" in data:
        sequences=data["sequence"]
        sequences_len=np.array(data["sequence_length"],np.int32)
    sequence_symbol=None
    if "sequence_symbol" in data:
        sequence_symbol=np.array(data["sequence_symbol"])
    ## setting multi-modal data
    vector_modal=[]
    vector_modal_name={}
    modal_names=["vector_modal","profeat","dragon","chemical_fp"]
    for name in modal_names:
        if name in data:
            vector_modal_name[name]=len(vector_modal)
            vector_modal.append(data[name])
    ## setting multi-graph data
    graph_index_list=None
    if "graph_index_list" in data:
        graph_index_list=data["graph_index_list"]

    # data num
    if adjs is not None:
        Num=len(adjs)
    else:
        Num=max([len(v) for v in vector_modal])

    all_data=dotdict({})
    all_data.features=features
    all_data.nodes=nodes
    all_data.adjs=adjs
    if labels is not None:
        all_data.labels=np.array(labels)
    else:
        all_data.labels=None
    all_data.mask_label=mask_label
    all_data.node_label=node_label
    all_data.mask_node_label=mask_node_label
    all_data.label_list=label_list
    all_data.num=Num
    all_data.sequences=sequences
    all_data.sequences_len=sequences_len
    all_data.sequence_symbol=sequence_symbol
    all_data.vector_modal=vector_modal
    all_data.enabled_node_nums=enabled_node_nums

    if config["shuffle_data"] and not prohibit_shuffle:
        print("[INFO] data_shuffle is done")
        all_data=shuffle_data(all_data)


    # info. data: constant value
    info=dotdict({})
    info.all_node_num=None
    if features is not None:
        ## features: #graphs x #nodes(graph) x #features
        info.feature_dim=features.shape[2]
        info.graph_node_num=features.shape[1]
        #info.all_node_num=data["node_num"]
        info.feature_enabled=True
    elif nodes is not None:
        ## nodes: #graphs x #nodes(graph)
        info.feature_dim=0
        info.graph_node_num=nodes.shape[1]
        info.all_node_num=data["node_num"]
        info.feature_enabled=False
    elif adjs is None:
        pass
    else:
        print("[ERROR] feature or node are required")
        raise DataLoadError("Please confirm input data (%s) and configuration"%(filename) )
    if sequences is not None:
        info.sequence_max_length=sequences.shape[1]
        info.sequence_symbol_num=data["sequence_symbol_num"]
    else:
        info.sequence_max_length=0
        info.sequence_symbol_num=0

    if adjs is not None:
        info.graph_num=len(adjs)
    else:
        info.graph_num=0
    info.adj_channel_num=adj_channel_num
    if labels is not None:
        if "label_dim" in data:
            info.label_dim=data["label_dim"]
        else:
            if len(labels.shape)>=2:
                info.label_dim=labels.shape[1]
            else:
                info.label_dim=1
        if adjs is not None:
            if labels.shape[0]!=info.graph_num:
                print("[ERROR] checking not [0 dim of labels] = [length of adjacency matrices]")
                print(">>info.graph_num: ",info.graph_num)
                print(">>labels.shape[0]: ",labels.shape[0])
        else:
            if labels.shape[0]!=Num:
                print("[ERROR] checking not [0 dim of labels] = [the number of data]")
                print(">>info.graph_num: ",info.graph_num)
                print(">>#data: ",labels.shape[0])

    elif node_label is not None:
        #node_label: graph_num x node_num x label_dim
        #print("node shape:",node_label.shape)
        info.label_dim=node_label.shape[2]
        print("[INFO] node centric mode")
    else:
        info.label_dim=None
        if "label_dim" in data:
            info.label_dim=data["label_dim"]
    if ( (features is None or features.shape[0]==info.graph_num)
            and (nodes is None or nodes.shape[0]==info.graph_num)
            and (nodes is None or len(adjs)==info.graph_num)):
        print("[OK] checking #graphs")
    else:
        print("[ERROR] checking not  [0 dim of features] = [0 dim of nodes] = [length of adjacency matrices]")
        print(">> ",info.graph_num)
        if features is not None:
            print(">> ",features.shape[0])
        if nodes is not None:
            print(">> ",nodes.shape[0])
        print(">> ",len(adjs))
        raise DataLoadError("Please confirm input data (%s)"%(filename) )

    # multi modal info
    info.vector_modal_dim=[]
    for modal in vector_modal:
        info.vector_modal_dim.append(modal.shape[1])
    info.vector_modal_name=vector_modal_name

    info.graph_index_list=graph_index_list
    #used for class weights
    if all_data["mask_label"] is not None and all_data["labels"] is not None:
        sum_all = np.sum(all_data["mask_label"],axis=0)
        sum_positive =  np.sum(all_data["labels"],axis=0)
        sum_negative = sum_all-sum_positive
        pos_weight_epsilon=0.01
        info.pos_weight= (sum_negative+pos_weight_epsilon) / (sum_positive+pos_weight_epsilon)
    if "class_weight" in data:
        info.class_weight=data["class_weight"]

        # for visualization of molecules
    if "mol_info" in data:
            info.mol_info = data["mol_info"]

    print("The number of graphs                   =",info.graph_num)
    print("Dimension of a feature                 =",info.feature_dim)
    print("The maximum number of nodes in a graph =",info.graph_node_num)
    print("The number of nodes in all graphs      =",info.all_node_num)
    print("Dimension of a label                   =",info.label_dim)
    print("The number of adj. matrices in a graph =",info.adj_channel_num)
    if graph_index_list is not None:
        print("The number of graph_index_lists         =",len(info.graph_index_list))

    return all_data,info

def split_data(all_data,valid_data_rate=0.2,indices_for_train_data=None,indices_for_valid_data=None):
    """Split data into two sets, usually for traing vs validation.
    Parameters
    ----------
    all_data : a docdict that contains data for graph convolution, generated with load_data().
    valid_data_rate : a number between 0 and 1. Ignored if indices_for_train_data and indices_for_valid_data are None.
    indices_for_train_data : list of indices used to extract train data.
    indices_for_valid_data : list of indices used to extract validation data.

    Returns
    -------
    train_data : a dotdict containing data organized the same way as in all_data. Use this for training.
    valid_data : same as above, for validation.
    """
    if "label_list" in all_data and all_data["label_list"] is not None:
        return split_label_list(all_data,valid_data_rate,indices_for_train_data,indices_for_valid_data)

    if indices_for_train_data is None or indices_for_valid_data is None:
        valid_num=int(all_data.num*valid_data_rate)
        train_num=all_data.num-valid_num
        indices=np.arange(all_data.num)
        np.random.shuffle(indices)
        indices_for_train_data=indices[:train_num]
        indices_for_valid_data=indices[train_num:all_data.num]

    valid_data=dotdict({})
    train_data=dotdict({})
    keys_for_split=all_data.keys()-{"num"}

    # Need to make sure members of the dict are np.array instead of list?
    for k in keys_for_split:
        if all_data[k] is None:
            valid_data[k]=None
            train_data[k]=None
        elif k=="vector_modal":
            valid_data[k]=[None]*len(all_data[k])
            train_data[k]=[None]*len(all_data[k])
            for j in range(len(all_data[k])):
                valid_data[k][j]=np.array([all_data[k][j][i] for i in indices_for_valid_data])
                train_data[k][j]=np.array([all_data[k][j][i] for i in indices_for_train_data])
        else:
            if isinstance(all_data[k],np.ndarray):
                valid_data[k]=all_data[k][indices_for_valid_data]
                train_data[k]=all_data[k][indices_for_train_data]
            else:
                valid_data[k]=np.array([all_data[k][i] for i in indices_for_valid_data])
                train_data[k]=np.array([all_data[k][i] for i in indices_for_train_data])
    valid_data.num=len(indices_for_valid_data)
    train_data.num=len(indices_for_train_data)
    return train_data, valid_data


def shuffle_label_list(data):
    if "label_list" in data and data["label_list"] is not None:
        np.random.shuffle(data.label_list[0])
def generate_negative_pair(label_list):
    if len(label_list)==1:

        label_list[0][:]
    if "label_list" in data and data["label_list"] is not None:
        np.random.shuffle(data.label_list[0])


def split_label_list(all_data,valid_data_rate=0.2,indices_for_train_data=None,indices_for_valid_data=None):
    """Split data into two sets, usually for traing vs validation.
    Parameters
    ----------
    all_data : a docdict that contains data for graph convolution, generated with load_data().
    valid_data_rate : a number between 0 and 1. Ignored if indices_for_train_data and indices_for_valid_data are None.
    indices_for_train_data : list of indices used to extract train data.
    indices_for_valid_data : list of indices used to extract validation data.

    Returns
    -------
    train_data : a dotdict containing data organized the same way as in all_data. Use this for training.
    valid_data : same as above, for validation.
    """
    print(">>",len(all_data.label_list))
    print(">>",len(all_data.label_list[0]))
    print(">>",len(all_data.label_list[0][0]))
    if indices_for_train_data is None or indices_for_valid_data is None:
        n=len(all_data.label_list[0])
        valid_num=int(n*valid_data_rate)
        train_num=n-valid_num
        nid=np.array(list(range(n)))
        np.random.shuffle(nid)
        indices_for_train_data=nid[:train_num]
        indices_for_valid_data=nid[train_num:]

    valid_data=dotdict({})
    train_data=dotdict({})
    keys_for_split=all_data.keys()
    for k in keys_for_split:
        valid_data[k]= all_data[k]
        train_data[k]= all_data[k]
    train_data["label_list"]= all_data["label_list"][:,indices_for_train_data,:]
    valid_data["label_list"]= all_data["label_list"][:,indices_for_valid_data,:]
    return train_data, valid_data


