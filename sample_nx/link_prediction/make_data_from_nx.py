import networkx as nx
import numpy as np
G = nx.karate_club_graph()
#G = nx.read_edgelist('sample.txt', nodetype=int)
print("#nodes: ",nx.number_of_nodes(G))
print("#edges: ",nx.number_of_edges(G))
node_num=nx.number_of_nodes(G)
adj_idx=np.array(G.edges())
adj_val=np.ones((nx.number_of_edges(G),))
adj_shape=[node_num,node_num]
adjs=[(adj_idx,adj_val,adj_shape)]

node_ids=[G.nodes()]
# generating labels(edge) at random
label_list=[]
for _ in range(10):
    x1=np.random.randint(node_num)
    x2=np.random.randint(node_num)
    x3=np.random.randint(node_num)
    pos_label=[x1,0,x2]
    neg_label=[x1,0,x3]
    label_list.append(pos_label+neg_label)
label_list=np.array([label_list])
obj={
    "adj":[adjs],
    "node":node_ids,
    "node_num":node_num,
    "label_list":label_list,
    "max_node_num":node_num}

import joblib

joblib.dump(obj,"sample.jbl")

