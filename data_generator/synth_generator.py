#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import os

from edward.models import Bernoulli, Normal
from edward.util import Progbar
from keras.layers import *
from keras import backend as K
from scipy.misc import imsave

K.set_learning_phase(0)
ed.set_seed(42)


M = 100  # batch size during training
D = 15 # the number of nodes
d = 1 # latent dimension

k1=np.zeros([M, D, d],dtype=np.float32)
k2=np.zeros([M, D, d],dtype=np.float32)

tri_idx=np.where(np.reshape(np.tri(D),[-1])==1)[0]

for i in range(D):
    if i%2==0:
        k1[:,i,:]=-1.0
        k2[:,i,:]=1.0
    else:
        k2[:,i,:]=-1.0
        k1[:,i,:]=1.0
# MODEL
def simple_generator(x):
    z = Normal(loc=x, scale=tf.ones([M, D, d]))
    hidden = z.value()
    z1=hidden
    z2=tf.transpose(hidden,[0,2,1])
    alpha=0.5
    a=tf.matmul(z1,z2)
    a=tf.reshape(a,[-1,D*D])
    ua=tf.gather(a,tri_idx,axis=1)
    p=tf.sigmoid(ua)
    x = Bernoulli(probs=p)
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
x1=simple_generator(k1)
x2=simple_generator(k2)

sess=ed.get_session()
tf.global_variables_initializer().run()

sample1=x1.sample().eval()
adj1=tri2sym(sample1)
sample2=x2.sample().eval()
adj2=tri2sym(sample2)
adjs=np.concatenate([adj1,adj2])
print(adjs.shape)
l1=[0 for _ in range(adj1.shape[0])]
l2=[1 for _ in range(adj1.shape[0])]
labels=np.concatenate([l1,l2])
data=list(zip(adjs,labels))
n=len(data)
np.random.shuffle(data)
## output
os.makedirs("synthetic",exist_ok=True)
fp=open("synthetic/adj.txt","w")
for k in range(n):
    adj=data[k][0]
    for i in range(D):
        fp.write(",".join(map(str,adj[i,:])))
        fp.write("\n")
    fp.write("\n")
fp=open("synthetic/feature.txt","w")
Level=2
for k in range(n):
    l=data[k][1]
    for i in range(D):
        if Level==0:
            if l==0:
                fp.write("1,1,0\n")
            elif l==1:
                fp.write("1,0,1\n")
        elif Level==1:
            if i%3==0:
                fp.write("1,0,0\n")
            if i%3==1:
                fp.write("0,1,0\n")
            if i%3==2:
                fp.write("0,0,1\n")
        else:
            fp.write("1,1,1\n")

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



