import joblib
import sys
import re
import networkx as nx
import matplotlib.pylab as plt

G = nx.Graph()
graph_id=0
if len(sys.argv)>1:
    graph_id=int(sys.argv[1])
# load dataset
obj=joblib.load("dataset.jbl")
prog = re.compile(r'<http://lod4all.net/resource#variant_hub_(.*)>')

mapping={}
short_name={}
short_name_inv={}
for name, node_id in obj["node_name"].items():
    m = prog.match(name)
    if m:
        a=m.group(1)
        if a not in short_name:
            xid=len(short_name)
            s="hub_"+str(xid)
            short_name[a]=s
            short_name_inv[s]=a
        name=short_name[a]
    mapping[node_id]=name
# #node in graph
n=obj["adj"][graph_id][2][0]
for i in range(n):
    k=obj["node"][graph_id][i]
    a=mapping[k]
    if(a in short_name_inv):
        print(a,short_name_inv[a])
        G.add_node(a,weight=1.0)
    else:
        G.add_node(a,weight=0.1)

edge_labels = {}
for i,j in obj["adj"][graph_id][0]:
    k0=obj["node"][graph_id][i]
    k1=obj["node"][graph_id][j]
    a0=mapping[k0]
    a1=mapping[k1]

    G.add_edge(a0,a1,weight=5)
    edge_labels[(a0,a1)]=1
    #print((i,j))

pos = nx.spring_layout(G)
#pos = nx.spectral_layout(G)
#pos = nx.shell_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=100, node_color="w")
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_edge_labels(G, pos,edge_labels)
nx.draw_networkx_labels(G, pos ,font_size=8, font_color="r")

print("========")
print("#node:",G.number_of_nodes())
print("#edge:",G.number_of_edges())
print("answer:",obj["label"][graph_id])
if "graph_name" in obj:
    graph_name=obj["graph_name"][graph_id]
    print("graph_name:",graph_name)
    m = prog.match(graph_name)
    if m:
        a=m.group(1)
        if a in short_name:
            print("graph_name:",short_name[a])



plt.xticks([])
plt.yticks([])
plt.show()

