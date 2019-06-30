import numpy as np
import joblib

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


print("[LOAD] Domain.cancer.DT.input.tsv")
data={}
for line in open("Domain.cancer.DT.input.tsv"):
    arr=line.strip().split("\t")
    if len(arr)>5:
        k=arr[0]
        if k not in data:
            #if len(data)==100:
            #	break
            data[k]=[]
        link=dotdict({})
        link.s=arr[1]
        link.sv=arr[2]
        link.p=arr[3]
        link.o=arr[4]
        link.ov=arr[5]
        data[k].append(link)


print("[LOAD] Domain.cancer.DT.answer.tsv")
answer={}
for line in open("Domain.cancer.DT.answer.tsv"):
    arr=line.strip().split("\t")
    answer[arr[0]]=int(arr[1])

all_nodes={}
graph_nodes={}
for k,v in data.items():
    if k not in graph_nodes:
        graph_nodes[k]={}
    for link in v:
        ss=link.s#+"="+link.sv
        oo=link.o#+"="+link.ov
        if ss not in all_nodes:
            all_nodes[ss]=len(all_nodes)
        if oo not in all_nodes:
            all_nodes[oo]=len(all_nodes)
        #
        if ss not in graph_nodes[k]:
            graph_nodes[k][ss]=len(graph_nodes[k])
        if oo not in graph_nodes[k]:
            graph_nodes[k][oo]=len(graph_nodes[k])
maxN=0
for k,v in graph_nodes.items():
    if maxN<len(v):
        maxN=len(v)

labels=[]
adjs=[]
nodes=[]
graph_names=[]
for k,v in data.items():
    ans=np.zeros((2,),np.int32)
    ans[answer[k]]=1
    labels.append(ans)
    adj_val=[]
    adj_idx=[]
    n=len(graph_nodes[k])
    node_vec=np.zeros((maxN,),np.int32)
    for node_name,i in graph_nodes[k].items():
        node_vec[i]=all_nodes[node_name]
    for link in v:
        ss=link.s#+"="+link.sv
        oo=link.o#+"="+link.ov
        i=graph_nodes[k][ss]
        j=graph_nodes[k][oo]
        adj_idx.append([i,j])
        adj_val.append(1)
    adjs.append((np.array(adj_idx),np.array(adj_val),np.array((n,n))))
    nodes.append(node_vec)
    graph_names.append(k)
print("node_num:",len(all_nodes))
print("maxN:",maxN)
obj={"label":np.array(labels),"adj":adjs,"node":np.array(nodes),"node_num":len(all_nodes),
    "node_name":all_nodes,"graph_name":graph_names}
print("[SAVE] dataset.jbl")
joblib.dump(obj, 'dataset.jbl')


