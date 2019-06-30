import joblib
import sys
import re
import numpy as np

filename=None
if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    print("[ERROR] dataset")
    quit()
obj=joblib.load(filename)
print(obj.keys())
if "label" in obj:
    print(obj["label"])
    counter={}
    for el in obj["label"]:
        k=tuple(el)
        if k not in counter:
            counter[k]=0
        counter[k]+=1
    print(counter)
elif "label_sparse" in obj:
    counter={}
    l=np.array(obj["label_sparse"].todense(),dtype=np.int32)
    m=np.array(obj["mask_label_sparse"].todense(),dtype=np.int32)
    #m=m[:10000,:]
    #l=l[:10000,:]
    cnt_all=np.sum(m,axis=0)
    cnt_pos=np.sum(l*m,axis=0)
    ###
    print("======")
    print(m.shape)
    print(l.shape)
    print("======")
    print(m[0,:])
    print(l[0,:])
    print("======")
    print(cnt_all)
    print(cnt_pos)
    d=cnt_pos*1.0/cnt_all
    print(d)
    quit()
    ###
    for el in l:
        k=tuple(el)
        print(k)
        if k not in counter:
            counter[k]=0
        counter[k]+=1
    print(counter)



