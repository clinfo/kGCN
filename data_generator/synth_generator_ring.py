#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from scipy.misc import imsave



M = 100  # batch size during training
D = 10 # the number of nodes
d = 6 # latent dimension

k1=np.zeros([D,D],dtype=np.int32)
k2=np.zeros([D,D],dtype=np.int32)

for i in range(d):
    k1[i,i]=1
    if i<d-1:
        k1[i,i+1]=1
        k1[i+1,i]=1
k1[0,d-1]=1
k1[d-1,0]=1
for i in range(d-1):
    k2[i,i]=1
    if i<d-2:
        k2[i,i+1]=1
        k2[i+1,i]=1
k2[0,d-2]=1
k2[d-2,0]=1
print(k1)
print(k2)

# MODEL
def simple_generator(x):
    for i in range(D-d):
        for j in range(d):
            a=np.random.binomial(1,0.1)
            x[d+i,j]=a
            x[j,d+i]=a
    return x
# tri-angle to symmetric dense matrix
def tri2sym(x):
    adj=np.zeros([M,D*D],np.int32)
    adj[:,tri_idx]=x
    adj=np.reshape(adj,[-1,D,D])
    adj+=np.transpose(adj,[0,2,1])
    for i in range(D):
        adj[:,i,i]=1
    return adj

adj=[]
label=[]
for i in range(100): 
    x1=simple_generator(k1)
    x2=simple_generator(k2)
    adj.append(x1)
    adj.append(x2)
    label.append(0)
    label.append(1)
data=list(zip(adj,label))
n=len(data)
np.random.shuffle(data)
## output

fp=open("synthetic/adj.txt","w")
for k in range(n):
    adj=data[k][0]
    for i in range(D):
        fp.write(",".join(map(str,adj[i,:])))
        fp.write("\n")
    fp.write("\n")
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
fp=open("synthetic/label.txt","w")
for k in range(n):
    l=data[k][1]
    if l==0:
        fp.write("1,0\n")
    elif l==1:
        fp.write("0,1\n")
    else:
        print("[ERROR]")



