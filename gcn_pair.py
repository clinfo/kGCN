import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.logging as logging
import numpy as np
import joblib
import time
import json
import argparse
import importlib

## gcn project
#import model
import kgcn.layers
from gcn import dotdict,NumPyArangeEncoder
from kgcn.data_util import align_size,dense_to_sparse,high_order_adj,split_adj,normalize_adj,DataLoadError,load_data
from scipy.sparse import coo_matrix
from kgcn.core import EarlyStopping
from gcn import get_default_config

def construct_feed(batch_idx,placeholders,data,graph_index_list,batch_size,dropout_rate=0.0):
    adjs=data.adjs
    features=data.features
    nodes=data.nodes
    labels=data.labels
    mask_label=data.mask_label
    node_label=data.node_label
    mask_node_label=data.mask_node_label
    sequences=data.sequences
    sequences_len=data.sequences_len
    graph_index_list_length=max(len(l) for l in graph_index_list)

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
                temp_labels=np.zeros((batch_size,mask_node_label.shape[1],mask_node_label.shape[2]),dtype=np.float32)
                idx=[graph_index_list[batch_idx[b]][p] for b in range(len(batch_idx))]
                temp_labels[:len(idx),:]=mask_node_label[idx,:,:]
                feed_dict[pl]=temp_labels
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
        elif key=="epsilon":
                eps=np.random.standard_normal(pl.shape)
                feed_dict[pl]=eps

    return feed_dict


def train(sess,config):
    batch_size=config["batch_size"]
    learning_rate=config["learning_rate"]
    model = importlib.import_module(config["model.py"])
    all_data,info = load_data(config,filename=config["dataset"])
    placeholders = model.build_placeholders(info,batch_size=batch_size,adj_channel_num=info.adj_channel_num)
    _,prediction,cost,cost_sum,metrics = model.build_model(placeholders,info,batch_size=batch_size,adj_channel_num=info.adj_channel_num,embedding_dim=config["embedding_dim"])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #train_step = tf.train.MomentumOptimizer(learning_rate,0.01).minimize(cost)
    # Initialize session
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Train model
    all_list_num=len(info.graph_index_list)
    print("#graph_index list = ",all_list_num)
    print("#data            = ",all_data.num)

    data_idx=list(range(len(info.graph_index_list)))
    n=int(all_list_num*0.8)
    train_idx=data_idx[:n]
    train_num=len(train_idx)
    valid_idx=data_idx[n:]
    valid_num=len(valid_idx)
    eary_stopping=EarlyStopping(config)
    start_t = time.time()
    for epoch in range(config["epoch"]):#[range(FLAGS.epochs):
        np.random.shuffle(train_idx)
        # training
        itr_num=int(np.ceil(train_num/batch_size))
        training_cost =0
        training_correct_count =0
        for itr in range(itr_num):
            offset_b=itr*batch_size
            batch_idx=train_idx[offset_b:offset_b+batch_size]
            feed_dict=construct_feed(batch_idx,placeholders,all_data,info.graph_index_list,batch_size=batch_size,dropout_rate=0.5)
            # running parameter update with tensorflow
            out_prediction=sess.run([prediction], feed_dict=feed_dict)
            #print(out_prediction)
            _,out_cost_sum,out_metrics = sess.run([train_step,cost_sum,metrics], feed_dict=feed_dict)
            training_cost +=out_cost_sum
            training_correct_count +=out_metrics["correct_count"]
            #print(out_metrics["correct_count"])
            #print(batch_size)
        training_cost/=train_num
        training_accuracy=training_correct_count/train_num

        # validation
        itr_num=int(np.ceil(valid_num/batch_size))
        validation_cost =0
        validation_correct_count =0
        for itr in range(itr_num):
            offset_b=itr*batch_size
            batch_idx=valid_idx[offset_b:offset_b+batch_size]
            feed_dict=construct_feed(batch_idx,placeholders,all_data,info.graph_index_list,batch_size=batch_size)
            out_cost_sum,out_metrics=sess.run([cost_sum,metrics], feed_dict=feed_dict)
            validation_cost += out_cost_sum
            validation_correct_count +=out_metrics["correct_count"]
        validation_cost/=valid_num
        validation_accuracy=validation_correct_count/valid_num

        # check point
        save_path=None
        if (epoch)%config["save_interval"] == 0:
            # save
            save_path =  config["save_model_path"]+"/model.%05d.ckpt"%(epoch)
            saver.save(sess,save_path)
        # early stopping and printing information
        if eary_stopping.evaluate_validation(validation_cost,
                {"epoch":epoch,
                    "validation_accuracy":validation_accuracy,
                    "validation_cost":validation_cost,
                    "training_accuracy":training_accuracy,
                    "training_cost":training_cost,
                    "save_path":save_path}):
            break

    train_time = time.time() - start_t
    print("traing time:{0}".format(train_time) + "[sec]")

    # saving last model
    #save_path =  config["save_model_path"]+"/model.last.ckpt"
    if "save_model" in config and config["save_model"] is not None:
        save_path =  config["save_model"]
        print("[SAVE] ",save_path)
        saver.save(sess,save_path)
    # validation
    start_t = time.time()
    data_idx=list(range(all_data.num))
    itr_num=int(np.ceil(all_data.num/batch_size))
    validation_cost =0
    validation_correct_count =0
    prediction_data=[]
    for itr in range(itr_num):
        offset_b=itr*batch_size
        batch_idx=data_idx[offset_b:offset_b+batch_size]
        feed_dict=construct_feed(batch_idx,placeholders,all_data,batch_size=batch_size)
        out_cost_sum,out_metrics,out_prediction=sess.run([cost_sum,metrics,prediction], feed_dict=feed_dict)
        validation_cost += out_cost_sum
        validation_correct_count +=out_metrics["correct_count"]
        prediction_data.append(out_prediction)

    validation_cost/=all_data.num
    validation_accuracy=validation_correct_count/all_data.num
    print("final cost =",validation_cost)
    print("accuracy   =",validation_accuracy)
    train_time = time.time() - start_t
    print("infer time:{0}".format(train_time) + "[sec]")
    if "save_result_train" in config:
        filename=config["save_result_train"]
        save_prediction(filename,prediction_data)


if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)
    tf.set_random_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
            help='train/infer')
    parser.add_argument('--config', type=str,
            default=None,
            nargs='?',
            help='config json file')
    parser.add_argument('--save-config',
            default=None,
            nargs='?',
            help='save config json file')
    parser.add_argument('--no-config',
            action='store_true',
            help='use default setting')
    parser.add_argument('--model', type=str,
            default=None,
            help='model')
    parser.add_argument('--dataset', type=str,
            default=None,
            help='dataset')

    args=parser.parse_args()
    # config
    config=get_default_config()
    if args.config is None:
        pass
        #parser.print_help()
        #quit()
    else:
        print("[LOAD] ",args.config)
        fp = open(args.config, 'r')
        config.update(json.load(fp))
    # option
    if args.model is not None:
        config["load_model"]=args.model
    if args.dataset is not None:
        config["dataset"]=args.dataset
    # setup
    with tf.Graph().as_default():
    #with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # mode
            if args.mode=="train":
                train(sess,config)
            elif args.mode=="infer":
                infer(sess,config)
    if args.save_config is not None:
        print("[SAVE] ",args.save_config)
        fp=open(args.save_config,"w")
        json.dump(config,fp, indent=4)

