import os
import argparse
import joblib
import numpy as np

def read_inputs(args):
    data={}
    for filename in args.input:
        for line in open(filename):
            arr=line.strip().split("\t")
            if len(arr)==3:
                e=arr[1]
                if e not in data:
                    data[e]=[]
                data[e].append((arr[0],arr[1],arr[2]))
    return data

def standardize_non_directional_data(data):
    out_data={k:[] for k,_ in data.items()}
    for key,r_data in data.items():
        for e in r_data:
            if e[0]<e[2]:
                out_data[key].append(e)
            else:
                out_data[key].append((e[2],e[1],e[0]))
    return data

def stratified_cv_split(data,cv,shuffle=True):
    data_split={}
    for key,r_data in data.items():
        prev=0
        split=[]
        for i in range(cv-1):
            t=int((i+1)*len(data[key])/cv)
            split.append((prev,t))
            prev=t
        split.append((prev,len(data[key])))
        data_split[key]=split
        if shuffle:
            np.random.shuffle(r_data)
    
    out_data={key:[] for key,r_data in data.items()}
    for i in range(cv):
        for key,r_data in data.items():
            s,t=data_split[key][i]
            #print(s,t)
            fold_data=r_data[s:t]
            out_data[key].append(fold_data)
    return out_data

def get_one_fold_data(cv_data,valid_rate,fold_i):
    train_valid_data=[]
    test_data=[]
    for key,cv_obj in cv_data.items():
        for cv_index,r_data in enumerate(cv_obj):
            if cv_index==fold_i:
                #test
                test_data.extend(r_data)
            else:
                train_valid_data.extend(r_data)
    np.random.shuffle(train_valid_data)
    n=int(len(train_valid_data)*valid_rate)
    valid_data=train_valid_data[:n]
    train_data=train_valid_data[n:]
    return train_data,valid_data,test_data

def save(filename,data):
    print("[SAVE]",filename)
    fp=open(filename,"w")
    for el in data:
        fp.write(el[0])
        fp.write("\t")
        fp.write(el[1])
        fp.write("\t")
        fp.write(el[2])
        fp.write("\n")

def build_adjs(data,node_mapping,edge_mapping,with_swap=True,with_self=True):
    node_num=len(node_mapping)
    enc_data={}
    for el in sorted(data):
        h=node_mapping[el[0]]
        r=edge_mapping[el[1]]
        t=node_mapping[el[2]]
        if r not in enc_data:
            enc_data[r]=[]
        enc_data[r].append((h,t))
    adjs=[]
    for r,pairs in enc_data.items():
        idx_list=[]
        for pair in pairs:
            idx_list.append((h,t))
            if with_swap:
                idx_list.append((t,h))
        if with_self:
            idx_list.append((t,h))
        idx_list=list(set(idx_list))
        adj_idx=[]
        adj_val=[]
        for pair in sorted(idx_list):
            h,t=pair
            adj_idx.append([h,t])
            adj_val.append(1)
        adj=(np.array(adj_idx),np.array(adj_val),np.array((node_num,node_num)))
        adjs.append(adj)
    return adjs

def build_set(data,node_mapping,edge_mapping,with_swap=True,with_self=True):
    hr_t_set={}
    r_ht_set={}
    for el in data:
        h=node_mapping[el[0]]
        r=edge_mapping[el[1]]
        t=node_mapping[el[2]]
        ##
        key=(h,r)
        if key not in hr_t_set:
            hr_t_set[key]=set()
        hr_t_set[key].add(t)
        ##
        key=r
        if key not in r_ht_set:
            r_ht_set[key]=set()
        r_ht_set[key].add(h)
        r_ht_set[key].add(t)
    
    hr_t_set={k:list(v) for k,v in hr_t_set.items()}
    r_ht_set={k:list(v) for k,v in r_ht_set.items()}
    return hr_t_set,r_ht_set

def build_label_list(data,node_mapping,edge_mapping,negative_label=True,target_edge=None):
    label_list=[]
    neg_data={}
    neg_data_cnt={}
    if negative_label:
        hr_t_set,r_ht_set = build_set(data,node_mapping,edge_mapping)
        for k,v in r_ht_set.items():
            neg_data[k]=np.random.choice(v,len(data))
            neg_data_cnt[k]=0
    for el in data:
        if target_edge is None or target_edge==el[1]:
            h=node_mapping[el[0]]
            r=edge_mapping[el[1]]
            t=node_mapping[el[2]]
            if negative_label:
                x=neg_data[r][neg_data_cnt[k]]
                neg_data_cnt[k]+=1
                l=[h,r,t,h,r,x]
            else:
                l=[h,r,t,0,0,0]
            label_list.append(l)
    return label_list
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
        nargs='*',
        default=[],
        type=str)
    parser.add_argument('--output',
        default="./data/",
        type=str)
    parser.add_argument('--output_txt',
        action="store_true",
        help='')
    parser.add_argument('--non-directional',
        action="store_true",
        help='')
    parser.add_argument('--cv',
        default=5,
        type=int)
    parser.add_argument('--valid_rate',
        default=0.2,
        type=float)
    parser.add_argument('--target_edge',
        default=None,
        type=str)
    args=parser.parse_args()
    
    ###
    data=read_inputs(args)
    total_num=0
    for r,r_data in data.items():
        total_num+=len(r_data)
        print(r,len(r_data))
    print("total:",total_num)
    ###
    if args.non_directional:
        print("... standardize non-directional data")
        data=standardize_non_directional_data(data)
    ###
    print("... remove duplicated edges")
    for k in data.keys():
        data[k]=list(set(data[k]))
    ###
    node_mapping={}
    for r,r_data in data.items():
        for e in r_data:
            if e[0] not in node_mapping:
                node_mapping[e[0]]=len(node_mapping)
            if e[2] not in node_mapping:
                node_mapping[e[2]]=len(node_mapping)
    edge_mapping={k:i for i,k in enumerate(data.keys())}
    ###
    print("... split data")
    cv_data=stratified_cv_split(data,args.cv,shuffle=True)
    for fold_i in range(args.cv):
        train_data,valid_data,test_data=get_one_fold_data(cv_data,args.valid_rate,fold_i)
        ###
        if args.output_txt:
            out_path=os.path.join(args.output,"fold"+str(fold_i))
            os.makedirs(out_path,exist_ok=True)
            filename=os.path.join(args.output,"fold"+str(fold_i),
                    "triplets-train.txt")
            save(filename,train_data)
            filename=os.path.join(args.output,"fold"+str(fold_i),
                    "triplets-valid.txt")
            save(filename,valid_data)
            filename=os.path.join(args.output,"fold"+str(fold_i),
                    "triplets-test.txt")
            save(filename,test_data)
        ###
        train_valid_data = train_data+valid_data
        adjs = build_adjs(train_valid_data,node_mapping,edge_mapping,with_swap=True,with_self=True)
        label_list = build_label_list(train_valid_data,node_mapping,edge_mapping,negative_label=False,
                target_edge=args.target_edge)
        test_label_list = build_label_list(test_data,node_mapping,edge_mapping,negative_label=True,
                target_edge=args.target_edge)
        dataset={
            "adj":[adjs],
            "node":np.array([list(range(len(node_mapping)))]),
            "node_num":len(node_mapping),
            "label_list":np.array([label_list]),
            "test_label_list":np.array([test_label_list]),
            "max_node_num":len(node_mapping)}
        out_path=os.path.join(args.output,"fold"+str(fold_i))
        os.makedirs(out_path,exist_ok=True)
        filename=os.path.join(args.output,"fold"+str(fold_i),
                "triplets.jbl")
        print("[SAVE]",filename)
        joblib.dump(dataset,filename)
    ###
    filename=os.path.join(args.output,"node_list.csv")
    fp=open(filename,"w")
    print("[SAVE]",filename)
    all_nodes_list=[None for i in range(len(node_mapping))]
    for k,v in node_mapping.items():
        all_nodes_list[v]=k
    for node in all_nodes_list:
        fp.write(node)
        fp.write("\n")


if __name__ == '__main__':
    main()
