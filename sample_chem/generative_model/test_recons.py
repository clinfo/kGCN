import joblib
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--num', type=int,
        default=10,
        help='#data')
    args=parser.parse_args()



    o=joblib.load("dataset.jbl")
    print(o.keys())
    feat=o["feature"][:10]
    adjs=[]
    for a in o["adj"][:10]:
        aa=np.zeros(a[2],dtype=np.float32)
        for i,v in enumerate(a[1]):
            aa[a[0][i][0],a[0][i][1]]=v
        adjs.append([aa])
    test_recons={"feature":feat,"dense_adj":adjs}

    joblib.dump(test_recons,"test_recons.jbl")

