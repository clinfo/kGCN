#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os



M = 100  # datasize
D = 10 # the number of nodes
p = 6 # ring size
q = 5 # ring size

k1=np.zeros([D,D],dtype=np.int32)
k2=np.zeros([D,D],dtype=np.int32)

# adj. matrix for a p-size ring
for i in range(p):
    k1[i,i]=1
    if i<p-1:
        k1[i,i+1]=1
        k1[i+1,i]=1
k1[0,p-1]=1
k1[p-1,0]=1
# adj. matrix for a q-size ring
for i in range(q):
    k2[i,i]=1
    if i<q-1:
        k2[i,i+1]=1
        k2[i+1,i]=1
k2[0,q-1]=1
k2[q-1,0]=1
print(k1)
print(k2)

# MODEL
def simple_generator(x,ring_size):
    # added noise nodes
    for i in range(D-ring_size):
        for j in range(ring_size):
            a=np.random.binomial(1,0.1)
            x[ring_size+i,j]=a
            x[j,ring_size+i]=a
    return x

adj=[]
label=[]
for i in range(100): 
    x1=simple_generator(k1,p)
    x2=simple_generator(k2,q)
    adj.append(x1)
    adj.append(x2)
    label.append(0)
    label.append(1)
data=list(zip(adj,label))
n=len(data)
np.random.shuffle(data)

## output
os.makedirs("synthetic",exist_ok=True)
## adj. output
print("[SAVE] synthetic/adj.txt")
fp=open("synthetic/adj.txt","w")
for k in range(n):
    adj=data[k][0]
    for i in range(D):
        fp.write(",".join(map(str,adj[i,:])))
        fp.write("\n")
    fp.write("\n")

# systhezizing node features with noise 
print("[SAVE] synthetic/feature.txt")
fp=open("synthetic/feature.txt","w")
Level=1
for k in range(n):
    l=data[k][1]
    for i in range(D):
        if Level==0:
            if l==0:
                fp.write("1,1,0\n")
            elif l==1:
                fp.write("1,0,1\n")
        else:
            if i%3==0:
                fp.write("1,0,0\n")
            if i%3==1:
                fp.write("0,1,0\n")
            if i%3==2:
                fp.write("0,0,1\n")

    fp.write("\n")

## label output
print("[SAVE] synthetic/label.txt")
fp=open("synthetic/label.txt","w")
for k in range(n):
    l=data[k][1]
    if l==0:
        fp.write("1,0\n")
    elif l==1:
        fp.write("0,1\n")
    else:
        print("[ERROR]")



