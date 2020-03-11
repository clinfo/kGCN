import glob
"""
names=[]
dataname="data.tsv"
for line in open(dataname):
    if len(line.strip())>0:
        arr=line.strip().split("\t")
        if int(arr[1])==1:
            arr=line.strip().split("\t")
            names.append(arr[0])
print(names)
"""
filename="protein.fa"
flag=False
id_list=[]
for i,line in enumerate(open(filename)):
    if len(line.strip())==0:
        seq_symbol=""
        flag=False
    elif line[0]==">":
        arr=line.strip().split(" ")
        if arr[1]=="1":
            flag=True
            id_list.append(arr[0])
        if flag:
            print(line.strip())
    else:
        if flag:
            print(line.strip())
        #seq_symbol+=line.strip()

import shutil

print(id_list)
for i,pid in enumerate(id_list):
    filelist=[name for name in glob.glob("viz_cnn/mol_%04d_*_embedded_layer_IG*"%(i,))]
    if len(filelist)==1:
        pid=pid.strip(">")
        dst="viz_cnn/%02d"+pid+".png"
        dst=dst%(i,)
        shutil.copy(filelist[0],dst)
        print(filelist[0],"=>",dst)

