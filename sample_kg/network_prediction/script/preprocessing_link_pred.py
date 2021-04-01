import numpy as np
import csv
import os, sys
import json
import joblib
import argparse

def load_graph(filenames, labels):
    edges = set()
    nodes = set()

    for filename in filenames:
        print("[LOAD]", filename)
        temp_base, ext = os.path.splitext(filename)
        base, data_type = os.path.splitext(temp_base)
        if data_type==".graph" or ext==".sif":
            for line in open(filename):
                arr = line.strip().split("\t")
                if len(arr)==2: ## where graphfile->2columns; node-node
                    edgetype = "interaction"
                    if edgetype not in labels:
                        labels[edgetype] = len(labels)
                    if arr[0]!=arr[1]: ##arr[0]=node1, arr[1]=node2
                        # edges.add((arr[0], "", arr[1]))
                        edges.add((arr[0], edgetype, arr[1]))
                    else:
                        print("[skip self-loop]", arr[0])
                    nodes.add(arr[0])
                    nodes.add(arr[1])
                elif len(arr)==3: ## where graphfile->3columns; node-edgetype-node
                    if arr[1] not in labels:
                        labels[arr[1]] = len(labels) # add label into labels dic
                    if arr[0]!=arr[2]:
                        #edges.add((arr[0],arr[2]))
                        edges.add((arr[0], arr[1], arr[2]))
                    else:
                        print("[skip self-loop]",arr[0])
                    nodes.add(arr[0])
                    nodes.add(arr[2])
                else:
                    print("[ERROR] unknown format")
    return edges, nodes, labels

def sample_neg_list(target_nodes, train_target_edges, n):
    neg_label_list = []
    i_list = np.random.choice(target_nodes, n) # pick nodes(#n) randomly
    j_list = np.random.choice(target_nodes, n)
    s = set(train_target_edges)
    for i,j in zip(i_list, j_list):
        if (i,0,j) not in s: # generate negative label
            neg_label_list.append((i,0,j))
    #r_list=np.random.choice(np.arange(2,len(labels)),n)
    #for i,j,r in zip(i_list,j_list,r_list):
    #	if (i,r,j) not in train_target_edges:
    #		neg_label_list.append((i,r,j))
    return neg_label_list #[(node, 0, node)...]

def build_label_list(target_nodes, train_target_edges, m):
    label_list=[]
    pi=0
    ni=0
    neg_sample=100
    neg_label_list = [None]
    for i in range(m):
        if i%len(neg_label_list)==0:
            neg_label_list = sample_neg_list(target_nodes, train_target_edges, neg_sample)
            ni=0
        if pi==len(train_target_edges):
            np.random.shuffle(train_target_edges)
            pi=0
        pos=train_target_edges[pi]
        neg=neg_label_list[ni]
        label_list.append(pos+neg)
        ni+=1
        pi+=1
    return label_list

def build_adjs(base_edges, self_edges, node_num):
    adj_idx=[]
    adj_val=[]

    train_graph_edges = set([(e[0],e[2]) for e in base_edges]) #((node index1, node index2), (node index3, node index4), ...)
    train_graph_edges = list(train_graph_edges) + [(e[0],e[2])for e in self_edges] # add selfloop
    for e in sorted(train_graph_edges):
        assert len(e)==2, "length mismatch"
        adj_idx.append([e[0], e[1]])
        adj_val.append(1)
    adjs = [(np.array(adj_idx), np.array(adj_val), np.array((node_num, node_num)))]
    return adjs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--basal', nargs='*', default=[], type=str, help='set basal graph')
    parser.add_argument('--train', nargs='*', default=[], type=str, help='set train dataset')
    parser.add_argument('--test', nargs='*', default=[], type=str, help='set test dataset')
    parser.add_argument('--output_jbl', type=str, default="dataset.jbl", help='save jbl file')
    parser.add_argument('--output_csv', type=str, default="dataset_node.csv", help='save csv file')
    parser.add_argument('--neg_sample', type=int, default=1000, help='sample')
    args=parser.parse_args()

    labels = {"negative":0, "self":1}
    base_edges, base_nodes, labels = load_graph(args.basal, labels) ## load base graph
    train_edges, train_nodes, labels = load_graph(args.train, labels) ## load train dataset
    test_edges, test_nodes, labels = load_graph(args.test, labels) ## load test dataset
    target_edges = train_edges|test_edges # target=train+test
    target_nodes = train_nodes|test_nodes

    all_edges = target_edges|base_edges
    all_nodes = target_nodes|base_nodes

    print("#non-target edges(basal):", len(base_edges))
    print("#target edges(train+test):", len(target_edges))
    print("#all edges:", len(all_edges))
    print("===")
    print("#non-target nodes(basal):", len(base_nodes))
    print("#target nodes(train+test):", len(target_nodes))
    print("#all nodes:", len(all_nodes))
    print("===")
    all_nodes_list = sorted(list(all_nodes))
    node_num = len(all_nodes_list)
    all_nodes_mapping = {el:i for i, el in enumerate(all_nodes_list)} # {'node1':0, 'node2':1, ...}
    #node_num=len(all_nodes)

    base_edges = [(all_nodes_mapping[e[0]], labels[e[1]], all_nodes_mapping[e[2]]) for e in base_edges] #[(node1 index, label index, node2 index), ...]
    train_edges = [(all_nodes_mapping[e[0]], labels[e[1]], all_nodes_mapping[e[2]]) for e in train_edges]
    test_edges = [(all_nodes_mapping[e[0]], labels[e[1]], all_nodes_mapping[e[2]]) for e in test_edges]
    target_edges = [(all_nodes_mapping[e[0]], labels[e[1]], all_nodes_mapping[e[2]]) for e in target_edges]
    all_edges = [(all_nodes_mapping[e[0]], labels[e[1]], all_nodes_mapping[e[2]]) for e in all_edges]
    self_edges = [(i, labels["self"], i) for i in range(len(all_nodes))]

    base_nodes = [all_nodes_mapping[e] for e in base_nodes]
    target_nodes = [all_nodes_mapping[e] for e in target_nodes]

    # np.random.shuffle(target_edges) # no need to shuffle here

    # target_edge_num = len(target_edges)
    # test_num = int(target_edge_num*0.2) # no need to generate test from target
    # train_target_edges=target_edges[:target_edge_num-test_num]
    # test_target_edges=target_edges[target_edge_num-test_num:]
    train_target_edges = train_edges # basal edges -> train
    test_target_edges = test_edges

    print("#train target edges:", len(train_target_edges))
    print("#test target edges:", len(test_target_edges))

    ## train label list
    m = len(train_target_edges)
    label_list = build_label_list(target_nodes, train_target_edges, m)
    ## test label list
    m = len(test_target_edges)
    test_label_list = build_label_list(target_nodes, test_target_edges, m)

    adjs = build_adjs(base_edges + train_edges, self_edges, node_num) #adjs=base+train+self
    node_ids = np.array([list(range(node_num))])
    max_node_num = node_num

    obj={
        "adj":adjs,
        "node":node_ids,
        "node_num":max_node_num,
        "label_list":np.array([label_list]),
        "test_label_list":np.array([test_label_list]),
        "max_node_num":max_node_num}

    print(obj)
    print("[SAVE]", args.output_jbl)
    joblib.dump(obj, args.output_jbl)
    fp=open(args.output_csv, "w")
    print("[SAVE]", args.output_csv)
    for node in all_nodes_list:
        fp.write(node)
        fp.write("\n")


