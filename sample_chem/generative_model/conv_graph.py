import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

import joblib
import argparse

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

import joblib
import argparse

def get_bond_type(k):
    if k==0:
        return Chem.rdchem.BondType.SINGLE
    elif k==1:
        return Chem.rdchem.BondType.DOUBLE
    elif k==2:
        return Chem.rdchem.BondType.TRIPLE
    elif k==3:
        return Chem.rdchem.BondType.AROMATIC
    else:
        return Chem.rdchem.BondType.UNSPECIFIED
## v = adj[:,i,j]
def get_edge(args, v):
    if args.multi:
        p=np.max(v)
        k=np.argmax(v)
        bond_type=get_bond_type(k)
        if not args.random_edge:
            return True,k,bond_type
        #print(p)
        enabled=False
        if np.random.rand()<p:
            enabled=True
        return enabled,k,bond_type
    else:
        if not args.random_edge:
            return True
        enabled=False
        if np.random.rand()<v:
            enabled=True
        return enabled
        

def sample_from_feature(vec):
    s=np.sum(vec)
    p=vec/s
    i=np.random.choice(list(range(len(p))),p=p)
    return i

def get_atom_features(args,feat, bool_id_feat=False, explicit_H=False):
    
    if args.random_atom_symbol:
        i=sample_from_feature(feat[0:44])
    else:
        i=np.argmax(feat[0:44])
    s=atom_symbols[i]
    
    i=np.argmax(feat[44:55])
    degrees=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d=degrees[i]
    
    i=np.argmax(feat[55:62])
    implicitValence=[0, 1, 2, 3, 4, 5, 6]
    iv=implicitValence[i]
    
    fc=feat[62]

    re=feat[63]

    hybridization=[
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
            SP3D, Chem.rdchem.HybridizationType.SP3D2
        ]
    i=np.argmax(feat[64:69])
    hy=hybridization[i]
    
    ar=feat[69]
    
    num_hs=[0, 1, 2, 3, 4]
    i=np.argmax(feat[70:75])
    hs=num_hs[i]
    
    return [s,d,iv,fc,re,hy,ar,hs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--output_dir',default="./", type=str)
    parser.add_argument('--threshold',default=0.9, type=float)
    parser.add_argument('--random_atom_num',action='store_true',
        help='atom num is generated at random')
    parser.add_argument('--random_atom_symbol',action='store_true',
        help='atom symbol is generated using atom features')
    parser.add_argument('--random_edge',action='store_true',
        help='atom edge is generated using edge probability')
    parser.add_argument('--num', type=int,
        default=10,
        help='#data')
    parser.add_argument(
        '-m', '--multi', action='store_true',
        help='multi-channel adj.')
    args=parser.parse_args()

    np.random.seed(1234)
    atom_symbols=['C','N','O','S','F','Si','P',
        'Cl','Br','Mg','Na','Ca','Fe','As','Al',
        'I','B','V','K','Tl','Yb','Sb','Sn',
        'Ag','Pd','Co','Se','Ti','Zn','H',  # H?
        'Li','Ge','Cu','Au','Ni','Cd','In','Mn',
        'Zr','Cr','Pt','Hg','Pb','Unknown']

    #o=joblib.load("recons.valid.jbl")
    #o=joblib.load("test_recons.jbl")
    o=joblib.load(args.input)
    print(o.keys())
    for index in range(args.num):
        print("... start",index)
        mol   = Chem.RWMol()
        adj= o["dense_adj"][index]
        if args.multi:
            connected_atoms=[]
            max_atom_num=o["feature"].shape[1]
            if args.random_atom_num:
                max_atom_num=np.random.randint(3,max_atom_num)
            for i in range(max_atom_num):
                if np.max(o["feature"][index,i,0:44])>0.01:
                    connected_atoms.append(i)
            ii=np.where(adj>args.threshold)
            print("#atoms :",len(connected_atoms))
            #connected_atoms=np.unique(np.concatenate([ii[1],ii[2]]))
            for i in connected_atoms:
                v = o["feature"][index,i,:]
                u=get_atom_features(args, v, bool_id_feat=False, explicit_H=False)
                #print(u[0])
                if u[0]!="Unknown":
                    atom=Chem.Atom(u[0])
                    idx = mol.AddAtom( atom )
            
            bonds=set()
            for i,j in zip(ii[1],ii[2]):
                i=int(i)
                j=int(j)
                if i < j and (i,j) not in bonds:
                    if i in connected_atoms and j in connected_atoms:
                        flag,k,bond_type=get_edge(args, adj[:,i,j])
                        if flag:
                            try:
                                mol.AddBond(i,j,bond_type)
                                bonds.add((i,j))
                            except:
                                print("[fail:bond]",i,j)
            print("#bonds :",len(bonds))
            try:
                if mol is not None:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
                    #smi=Chem.MolToSmiles(mol)
                    #mol=Chem.MolFromSmiles(smi)
                    #print(smi)
                    filename=args.output_dir+'image%05d.png'%(index,)
                    Draw.MolToFile(mol,filename)
                    print("[success%05d]"%(index,),filename)
            except:
                print("[fail%05d]"%(index,))
        else:
            ii=np.where(adj[0]>args.threshold)
            connected_atoms=np.unique(np.concatenate([ii[0],ii[1]]))
            for i in connected_atoms:
                v = o["feature"][index,i,:]
                u=get_atom_features(args, v, bool_id_feat=False, explicit_H=False)
                atom=Chem.Atom(u[0])
                idx = mol.AddAtom( atom )

            for i,j in zip(ii[0],ii[1]):
                bond_type=Chem.rdchem.BondType.UNSPECIFIED
                i=int(i)
                j=int(j)
                if i < j:
                    flag=get_edge(args, adj[0,i,j])
                    if flag:
                        mol.AddBond(i,j,bond_type)

            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
            smi=Chem.MolToSmiles(mol)
            mol=Chem.MolFromSmiles(smi)
            try:
                if mol is not None:
                    filename=args.output_dir+'image%05d.png'%(index,)
                    Draw.MolToFile(mol,filename)
                    print("[success%05d]"%(index,),filename)
            except:
                print("[fail%05d]"%(index,))

