import joblib
import sys
import re

graph_id=0
if len(sys.argv)>1:
    graph_id=int(sys.argv[1])
obj=joblib.load("dataset.jbl")
print(obj.keys())
v=obj["node"][graph_id]
print(v.tolist())
print(obj["label"][graph_id])

prog = re.compile(r'(.*)=(.*)')
values={}
for name, node_id in obj["node_name"].items():
    #print(name,node_id)
    m = prog.match(name)
    if m:
        k=m.group(1)
        v=m.group(2)
        if k not in values:
            values[k]={}
        if v not in values[k]:
            values[k][v]=node_id

for k,vs in values.items():
    print(k,len(vs))
    if(len(vs)>10):
        print(vs)
