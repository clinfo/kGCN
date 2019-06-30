import numpy as np
import os
import sys
import joblib
import argparse
import graph_tool as gt
#import graph_tool.all as gt

def dataset2graphset(obj):
    graph_db=[]
    labels=obj["label"]
    features=None
    node_labels=None
    if "feature" in obj:
        features=obj["feature"]
    if "node" in obj:
        node_labels=obj["node"]

    if "dense_adj" in obj:
        adjs=obj["dense_adj"]
        for k,adj in enumerate(adjs):
            ## making Graph
            g = gt.Graph(directed=False)
            m=len(adj)
            g.add_vertex(m)
            for i in range(m):
                for j in range(i+1):
                    el=adj[i][j]
                    if int(el)!=0:
                        g.add_edge(g.vertex(i), g.vertex(j))
            #a = gt.adjacency(g)
            #print(a)

            ## making feature
            if features is not None:
                f=features[k]
                g.vp.na = g.new_vertex_property("vector<float>")
                for i,v in enumerate(g.vertices()):
                    g.vp.na[v] = f[i]
            else:
                g.vp.nl = g.new_vertex_property("int")
                for i,v in enumerate(g.vertices()):
                    g.vp.nl[v] = node_labels[i]
                #dim_attributes = len(g.vp.na[i])
                #print(dim_attributes)
            graph_db.append(g)
    elif "adj" in obj:
        adjs=obj["adj"]
        for k,adj in enumerate(adjs):
            edges=adj[0]
            vals=adj[1]
            size=adj[2]
            m=size[0]
            g = gt.Graph(directed=False)
            g.add_vertex(m)
            es=edges[edges[:,0]>=edges[:,1]]
            g.add_edge_list(es)
            ## making feature
            if features is not None:
                f=features[k]
                g.vp.na = g.new_vertex_property("vector<float>")
                for i,v in enumerate(g.vertices()):
                    g.vp.na[v] = f[i]
            elif node_labels is not None:
                nl=node_labels[k]
                g.vp.nl = g.new_vertex_property("int",vals=nl[:m])
                g.vp.na = g.new_vertex_property("vector<float>",vals=[])
            graph_db.append(g)
    graph_obj={"graph":graph_db,"label":labels}
    return graph_obj

def concat_graphset(obj1,obj2):
    g1=obj1["graph"]
    l1=obj1["label"]
    g2=obj2["graph"]
    l2=obj2["label"]
    # length
    length=None
    if len(g1)==len(l1) and len(g2)==len(l2):
        length=(len(l1),len(l2))
    else:
        print("[ERROR] length mismatch")
    # concat
    g1.extend(g2)
    l1=np.concatenate((l1,l2),axis=0)
    return {"graph":g1,"label":l1},length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "graph kernel file")
    parser.add_argument("input", help = "input file path", type = str)
    parser.add_argument("--output","-o",default="./sample.graph.jbl", help = "output joblib file path", type = str)

    args = parser.parse_args()

    print("[LOAD] ",args.input)
    obj=joblib.load(args.input)
    graph_obj=dataset2graphset(obj)
    print("[SAVE] ",args.output)
    joblib.dump(graph_obj,args.output)

#g=graph_db[0]
#a = gt.adjacency(g)
#print(a)
#print([g.vp.nl[v] for v in g.vertices()])

