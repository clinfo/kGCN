import argparse
import glob
import os
import pathlib
import random
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from sklearn.preprocessing import LabelEncoder
from tensorflow.python_io import TFRecordWriter
from tensorflow.train import Feature, Features, FloatList, Int64List, Example
import joblib

#import oddt.toolkits.extras.rdkit as ordkit
from mendeleev import element

ELECTRONEGATIVITIES = [element(i).electronegativity('pauling') for i in range(1, 100)]
ELECTRONEGATIVITIES_NO_NONE = []
for e in ELECTRONEGATIVITIES:
    if e is None:
        ELECTRONEGATIVITIES_NO_NONE.append(0)
    else:
        ELECTRONEGATIVITIES_NO_NONE.append(e)


def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
    )
    parser.add_argument(
        '-l', '--label', default=None, type=str,
        help='help'
    )
    parser.add_argument(
        '--label_dim', default=None, type=int,
        help='help'
    )
    parser.add_argument(
        '-s', '--smarts', default=None, type=str,
        help='help'
    )
    parser.add_argument(
        '--sdf', default=None, type=str,
        help='help'
    )
    parser.add_argument(
        '--sdf_dir', default=None, type=str,
        help='help'
    )
    parser.add_argument(
        '--assay_dir', default=None, type=str,
        help='help'
    )
    parser.add_argument(
        '--assay_num_limit', default=None, type=int,
        help='help'
    )
    parser.add_argument(
        '--assay_pos_num_limit', default=None, type=int,
        help='help'
    )
    parser.add_argument(
        '--assay_neg_num_limit', default=None, type=int,
        help='help'
    )
    parser.add_argument(
        '--sparse_label', action='store_true', default=False,
        help='help'
    )
    parser.add_argument(
        '-a', '--atom_num_limit', type=int,
        help='help'
    )
    parser.add_argument(
        '--no_header', action='store_true', default=False,
        help='no header line in the label file'
    )
    parser.add_argument(
        '--without_mask', action='store_true', default=False,
        help='without label mask'
    )
    parser.add_argument(
        '-o', '--output', default="dataset.jbl", type=str,
        help='help'
    )
    parser.add_argument(
        '--vector_modal', default=None, type=str,
        help='vector modal csv'
    )
    parser.add_argument(
        '--sdf_label', default=None, type=str,
        help='property name used as labels'
    )
    parser.add_argument(
        '--sdf_label_active', default="Active", type=str,
        help='property name used as labels'
    )
    parser.add_argument(
        '--sdf_label_inactive', default="Inactive", type=str,
        help='property name used as labels'
    )
    parser.add_argument(
        '--solubility', action='store_true', default=False,
        help='solubilites in SDF as labels'
    )
    parser.add_argument(
        '--csv-reaxys', default=None, type=str,
        help='path to a csv containing reaxys data.'
    )
    parser.add_argument(
        '--multimodal', action='store_true', default=False,
        help='help'
    )
    parser.add_argument(
        '--no_pseudo_negative', action='store_true', default=False,
        help='help'
    )
    parser.add_argument(
        '--max_len_seq', type=int, default=None,
        help='help'
    )
    parser.add_argument(
        '--generate_mfp', action='store_true', default=False,
        help='generate Morgan Fingerprint using RDkit'
    )
    parser.add_argument(
        '--use_sybyl', action='store_true', default=False,
        help='[Additional features] SYBYL atom types'
    )
    parser.add_argument(
        '--use_electronegativity', action='store_true', default=False,
        help='[Additional features] electronegativity'
    )
    parser.add_argument(
        '--use_gasteiger', action='store_true', default=False,
        help='[Additional features] gasteiger charge'
    )
    parser.add_argument(
        '--degree_dim', type=int,
        default=17,
        help='[Additional features] maximum number of degree'
    )
    parser.add_argument(
        '--use_deepchem_feature', action='store_true', default=False,
        help='75dim used in deepchem default'
    )
    return parser.parse_args()


def generate_inactive_data(label_data, label_mask):
    data_index = np.argwhere(label_mask == 1)
    neg_count=0
    pos_count=0
    for data_point in data_index:
        if label_data[tuple(data_point)] == 0:
            neg_count+=1
        else:
            pos_count+=1
    print("active count:",pos_count)
    print("inactive count:",neg_count)
    if pos_count>neg_count:
        actives = np.argwhere(label_data == 1)
        np.random.shuffle(actives[:, 1])  # in place
        count=0
        for inactive_data_point in actives:
            if label_data[tuple(inactive_data_point)] == 0:
                label_mask[tuple(inactive_data_point)] = 1
                count+=1
        print("pseudo inactive count:",count)

def generate_multimodal_data(args, mol_obj_list, label_data, label_mask, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat):
    """make inactive data with mol data and protein data & count active = inactive, inactive = over300000
        
    Arguments:
        negative_generate --make inactive data from protein,compound,label data & use index from active data

        Returns:#Yields:
                negative_adj,feature,seq,seq_symbol,drogon_data
    """
    ##make inactives
    if not args.no_pseudo_negative:
        enabled_mol_index,enabled_task_index=np.where(label_mask==1)
        active_count = np.where(label_data[enabled_mol_index,enabled_task_index] == 1)[0].shape[0]
        inactive_count = np.where(label_data[enabled_mol_index,enabled_task_index] == 0)[0].shape[0]
        make_count = active_count - inactive_count
        print("[multimodal] count=",len(enabled_mol_index))
        print("[multimodal] active count=",active_count)
        print("[multimodal] inactive count=",inactive_count)
        print("[multimodal] pseudo inactive count=",make_count)
        print("[multimodal] #mols: ",len(mol_obj_list))
        print("[multimodal] #proteins: ",len(task_name_list))
        if make_count+active_count+inactive_count> len(mol_obj_list)*len(task_name_list):
            print("[WARN] all of the rest data are pseudo negative!")
            negative_data_index=np.where(label_mask==0)
            label_mask[label_mask==0]=1
        else:
            negative_count=0
            negative_data_index=[[],[]]
            while(negative_count<make_count):
                x=np.where(label_mask==1)
                mol_index = np.random.randint(0,len(mol_id_list),make_count-negative_count)
                protein_index = np.random.randint(0,len(task_name_list),make_count-negative_count)
                flags=label_mask[mol_index,protein_index]
                new_mol_index=mol_index[flags==0]
                new_protein_index=protein_index[flags==0]
                if len(new_mol_index)>0:
                    label_mask[new_mol_index,new_protein_index]=1
                    label_data[new_mol_index,new_protein_index]=0 # negative
                    new_index=np.unique(np.array([new_mol_index,new_protein_index]),axis=1)
                    negative_data_index=np.concatenate([negative_data_index,new_index],axis=1)
                    negative_count+=new_index.shape[1]
                    print("#negative count:",negative_count)
                    x=np.where(label_mask==1)

    #convert_multimodal_label
    x=np.where(label_mask==1)
    print("#data",x[0].shape)
    ll=label_data[x[0],x[1]]
    if dragon_data is not None:
        dragon_data=dragon_data[x[0]]
    if mol_obj_list is not None:
        mol_obj_list=np.array(mol_obj_list)[x[0]]
    if seq is not None:
        seq=seq[x[1]]
        seq_symbol=seq_symbol[x[1]]
    if profeat is not None:
        profeat=profeat[x[1]]
    max_label=np.max(ll)
    print("[multimodal] maximum label",max_label)
    if args.label_dim is None:
        label_dim=int(max_label)+1
    else:
        label_dim = args.label_dim
    print("[multimodal] label dim.",label_dim)
    if label_dim<=2:
        new_label_data=np.zeros((ll.shape[0],2))
        new_label_data[ll==1,1]=1
        new_label_data[ll==0,0]=1
        new_mask_label=np.ones_like(new_label_data)
    else:
        new_label_data=np.zeros((ll.shape[0],label_dim))
        print(ll)
        for i,l in enumerate(ll):
            new_label_data[i,int(l)]=1
        new_mask_label=np.ones_like(new_label_data)

    return mol_obj_list, new_label_data, new_mask_label, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat

def dense_to_sparse(dense):
    from scipy.sparse import coo_matrix
    coo = coo_matrix(dense)
    sh = coo.shape
    val = coo.data
    sp = list(zip(coo.row, coo.col))
    return np.array(sp), np.array(val, dtype=np.float32), np.array(sh)

def read_label_file(args):
    file = args.label
    if file is None:
        return None, None, None
    _, ext = os.path.splitext(file)
    sep = "\t" if ext == ".txt" else ","
    csv = pd.read_csv(file, header=None, delimiter=sep) if args.no_header else pd.read_csv(file, delimiter=sep)

    header = csv.columns.tolist()
    print("label name: ", header)

    label = np.array(csv.values[:, 1], dtype=np.float32) if ext == ".txt" else np.array(csv.values, dtype=np.float32)
    # Convert nan to mask
    mask_label = np.zeros_like(label, dtype=np.float32)
    mask_label[~np.isnan(label)] = 1
    return header, label, mask_label

def read_profeat():
    dict_profeat={}
    filename="profeat.txt"
    if os.path.exists(filename):
        for line in open(filename):
            arr=line.split("\t")
            dict_profeat[arr[0]]=list(map(float,arr[1:]))
        return dict_profeat
    else:
        return None

# Get the information from atom
def atom_features(atom, explicit_H=False, use_sybyl=False, use_electronegativity=False, use_gasteiger=False,degree_dim=11):
    if use_sybyl:
        import oddt.toolkits.extras.rdkit as ordkit
        atom_type = ordkit._sybyl_atom_type(atom)
        atom_list = ['C.ar', 'C.cat', 'C.1', 'C.2', 'C.3', 'N.ar', 'N.am', 'N.pl3', 'N.1', 'N.2', 'N.3', 'N.4', 'O.co2',
                     'O.2', 'O.3', 'S.O', 'S.o2', 'S.2', 'S.3', 'F', 'Si', 'P', 'P3', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                     'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
                     'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In','Mn', 'Zr', 'Cr', 'Pt',
                     'Hg', 'Pb', 'Unknown']
    else:
        atom_type = atom.GetSymbol()
        atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                     'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
                     'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In','Mn', 'Zr', 'Cr', 'Pt',
                     'Hg', 'Pb', 'Unknown']


    results = one_of_k_encoding_unk(atom_type, atom_list) + \
              one_of_k_encoding(atom.GetDegree(), list(range(degree_dim))) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(),
                                    [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                     Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                     Chem.rdchem.HybridizationType.SP3D2]) + \
              [atom.GetIsAromatic()]
    if use_electronegativity:
        results = results + [ELECTRONEGATIVITIES_NO_NONE[atom.GetAtomicNum() - 1]]

    if use_gasteiger:
        gasteiger = atom.GetDoubleProp('_GasteigerCharge')
        if np.isnan(gasteiger) or np.isinf(gasteiger):
            gasteiger = 0 # because the mean is 0
        results = results + [gasteiger]


    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)


# Convert to one-hot expression
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class AssayData:
    def __init__(self):
        self.path = ""
        self.df_assay = None
        self.dict_id_mol = None
        self.dragon_data = None
        self.seq = None
        self.seq_symbol = None
        self.profeat = None

    def _build_df_assay(self, args, assay_path):
        assay_filename = os.path.join(assay_path, "assay.csv")
        df_assay = pd.read_csv(assay_filename, sep='\t', header=None, names=['mol_id', assay_path], index_col=0)
        df_assay.replace('inactive', -1, inplace=True)
        df_assay.replace('active', 1, inplace=True)

        # reading sdf
        sdf_filename = os.path.join(assay_path, "SDF_wash/SDF_wash.sdf")
        if not os.path.exists(sdf_filename):
            print("[PASS] not found:", sdf_filename)
            return

        dict_id_mol, drop_index = {}, []
        for mol_id_label, mol in zip(df_assay.iterrows(), Chem.ForwardSDMolSupplier(sdf_filename)):
            mol_id = mol_id_label[0]
            if mol is None:
                drop_index.append(mol_id)
                continue
            # Skip the compound whose total number of atoms is larger than "atom_num_limit"
            if args.atom_num_limit is not None and mol.GetNumAtoms() > args.atom_num_limit:
                drop_index.append(mol_id)
                continue
            dict_id_mol[mol_id] = mol

        df_assay.drop(index=drop_index, inplace=True)
        self.dict_assay = df_assay.T.to_dict('list')
        self.dict_id_mol = dict_id_mol
        self.drop_index = drop_index

    def _build_dragon_data(self, assay_path):
        dragon_assay_filename1 = os.path.join(assay_path, "SDF_wash_dragon894.csv")
        dragon_assay_filename2 = os.path.join(assay_path, "Dragon_CGBVS.list")
        # reading csv
        df_dragon = None
        if os.path.exists(dragon_assay_filename1):
            df_dragon = pd.read_csv(dragon_assay_filename1, sep=',', header=None, index_col=0)
        elif os.path.exists(dragon_assay_filename2):
            df_dragon = pd.read_csv(dragon_assay_filename2, sep='\t', header=None, index_col=0)

        # deleting lines in drop_list and converting into dictionary
        self.dragon_data = None
        if df_dragon is not None:
            dragon_data = {}
            df_dragon.drop(index=self.drop_index, inplace=True)
            dragon_data = {x[0]: x[1:] for x in df_dragon.itertuples()}
            self.dragon_data = dragon_data

    def _build_seq(self, assay_path):
        seq_filename = os.path.join(assay_path, "protein.fa")
        seq, seq_symbol = None, None
        if os.path.exists(seq_filename):
            seq_symbol = ""
            for line in open(seq_filename):
                # skip comment line
                if line[0] != ">":
                    seq_symbol += line.strip()
            # to integer
            seq = [ord(x) - ord("A") for x in seq_symbol]
            # to float
            seq = list(map(float, seq))
        self.seq = seq
        self.seq_symbol = seq_symbol

    def _build_profeat(self, assay_path, dict_profeat):
        b=os.path.basename(assay_path)
        if dict_profeat is not None:
            if b in dict_profeat:
                v=dict_profeat[b]
                self.profeat=v
            elif dict_profeat is not None:
                print("[WARN]",b,"does not have profeat")
                self.profeat=None
            else:
                self.profeat=None

    def build_from_dir(self, args, assay_filename, dict_profeat=None):
        assay_path = os.path.dirname(assay_filename)
        self.path = assay_path
        print("[LOAD]", assay_path)
        self._build_df_assay(args, assay_path)
        self._build_dragon_data(assay_path)
        self._build_seq(assay_path)
        self._build_profeat(assay_path,dict_profeat)
        return self


def concat_assay(assay_list):
    dict_all_assay=None
    for assay_data in assay_list:
        if assay_data.dict_assay is not None:
            if dict_all_assay is None:
                dict_all_assay = {}
            dict_all_assay.update({(assay_data.path,str(k)): v for k, v in assay_data.dict_assay.items()})

    dict_all_id_mol = None
    for assay_data in assay_list:
        if assay_data.dict_id_mol is not None:
            if dict_all_id_mol is None:
                dict_all_id_mol = {}
            dict_all_id_mol.update({str(k): v for k, v in assay_data.dict_id_mol.items()})

    dict_dragon_data = None
    for assay_data in assay_list:
        if assay_data.dragon_data is not None:
            if dict_dragon_data is None:
                dict_dragon_data = {}
            dict_dragon_data.update({str(k): v for k, v in assay_data.dragon_data.items()})

    seq, seq_symbol = None, None
    for assay_data in assay_list:
        if assay_data.seq is not None and assay_data.seq_symbol is not None:
            assay_data.path
            if seq is None:
                seq, seq_symbol = {}, {}
            seq[assay_data.path]=assay_data.seq
            seq_symbol[assay_data.path]=assay_data.seq_symbol
    
    dict_profeat = None 
    for assay_data in assay_list:
        if assay_data.profeat is not None:
            if dict_profeat is None:
                dict_profeat = {}
            dict_profeat[assay_data.path]=assay_data.profeat
    
    return dict_all_assay, dict_all_id_mol, dict_dragon_data, seq, seq_symbol, dict_profeat


def summarize_assay(args, df_all_assay):
    # summarization
    df_count = df_all_assay.count()
    summary = [df_count]
    count_data = [1, -1]
    column_names = ["count","count_pos","count_neg"]
    #+ list(map(lambda x: "count_"+str(x), count_data))
    for el in count_data:
        summary.append((df_all_assay == el).sum())
    df_summary = pd.concat(summary, axis=1)
    # rename colomns name
    df_summary.columns = column_names
    basepath, ext = os.path.splitext(args.output)
    summary_filename = basepath + ".summary.csv"
    print("[SAVE]", summary_filename)
    df_summary.to_csv(summary_filename)
    return df_summary


def build_all_assay_data(args):
    assay_list = []
    dict_profeat=read_profeat()
    for assay_filename in glob.iglob(os.path.join(args.assay_dir, '**/assay.csv'), recursive=True):
        assay_list.append(AssayData().build_from_dir(args, assay_filename,dict_profeat=dict_profeat))

    dict_all_assay, dict_all_id_mol, dict_dragon_data, seq, seq_symbol, dict_profeat = concat_assay(assay_list)
    #
    assay_ids=np.unique([assay_id for assay_id,mol_id in dict_all_assay.keys()])
    mol_ids=np.unique([mol_id for assay_id,mol_id in dict_all_assay.keys()])
    ## building label table
    label_data=np.empty((len(mol_ids),len(assay_ids)),dtype=np.float32)
    label_data[:,:]=np.nan
    for i,mi in enumerate(mol_ids):
        for j,aj in enumerate(assay_ids):
            if (aj,mi) in dict_all_assay:
                label_data[i,j]=dict_all_assay[(aj,mi)][0]
    ## dropping limitted tasks
    dd=pd.DataFrame(label_data)
    df_summary = summarize_assay(args, dd)
    if args.assay_num_limit is not None:
        names = df_summary.query('count >= '+str(args.assay_num_limit)).index
        label_data = label_data[:,names]
        assay_ids=assay_ids[names]
        # dropping rows consisting of only nan data
        x=~np.all(np.isnan(label_data),axis=1)
        label_data=label_data[x,:]
        mol_ids=mol_ids[x]
    if args.assay_pos_num_limit is not None:
        names = df_summary.query('count_pos >= '+str(args.assay_pos_num_limit)).index
        label_data = label_data[:,names]
        assay_ids=assay_ids[names]
        # dropping rows consisting of only nan data
        x=~np.all(np.isnan(label_data),axis=1)
        label_data=label_data[x,:]
        mol_ids=mol_ids[x]
    if args.assay_neg_num_limit is not None:
        names = df_summary.query('count_neg >= '+str(args.assay_neg_num_limit)).index
        label_data = label_data[:,names]
        assay_ids=assay_ids[names]
        # dropping rows consisting of only nan data
        x=~np.all(np.isnan(label_data),axis=1)
        label_data=label_data[x,:]
        mol_ids=mol_ids[x]
    #
    #
    ##
    task_name_list = list(assay_ids)
    mol_id_list = list(mol_ids)
    if dict_all_id_mol:
        mol_list = [dict_all_id_mol[mi] for mi in mol_ids]
    dragon_data = None
    if dict_dragon_data:
        dragon_data = np.array([dict_dragon_data[mi] for mi in mol_ids])
    if seq:
        seq= np.array([seq[mi] for mi in task_name_list])
        seq_symbol= np.array([seq_symbol[mi] for mi in task_name_list])
    profeat_list=None
    if dict_profeat:
        profeat_list= np.array([dict_profeat[mi] for mi in task_name_list])
    #
    
    #label_data = np.array(df_all_assay.values, dtype=np.float32)
    mask_label = np.zeros_like(label_data, dtype=np.float32)
    mask_label[~np.isnan(label_data)] = 1
    label_data[np.isnan(label_data)] = 0
    label_data[label_data < 0] = 0
    return mol_list, label_data, mask_label, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat_list


def build_vector_modal(args):
    filename = args.vector_modal
    # reading csv
    _, ext = os.path.splitext(filename)
    if ext == ".des" or ext == ".txt":
        df = pd.read_csv(filename, sep='\t', index_col=0)
    else:
        df = pd.read_csv(filename)
    return df.values


def parse_csv(args):
    df = pd.read_csv(args.csv_reaxys)
    le = LabelEncoder()
    label_data = le.fit_transform(df['reaction_core'])
    with open("class.sma", 'w') as sma:
        sma.write("\n".join(list(le.classes_)))
    label_data = np.expand_dims(label_data, axis=1)
    mol_obj_list = (Chem.MolFromSmarts(product) for product in df['product'])
    label_mask = np.ones_like(label_data)
    publication_years = df['max_publication_year']
    return mol_obj_list, label_data, label_mask, publication_years


def extract_mol_info(args):
    dragon_data, task_name_list, seq, seq_symbol, profeat, publication_years = [], [], [], [], [], []
    if args.smarts is not None:
        _, label_data, label_mask = read_label_file(args)  # label name, label, valid/invalid mask of label
        with open(args.smarts, "r") as f:
            lines = f.readlines()
        mol_obj_list = [Chem.MolFromSmarts(line) for line in lines]
    elif args.sdf_dir is not None:
        filename = os.path.join(args.sdf_dir, "SDF_wash.sdf")
        if not os.path.exists(filename):
            print("[PASS] not found:", filename)
        mol_obj_list = [mol for mol in Chem.SDMolSupplier(filename)]
        _, label_data, label_mask = read_label_file(args)  # label name, label, valid/invalid mask of label
    elif args.sdf is not None:
        filename = args.sdf
        if not os.path.exists(filename):
            print("[PASS] not found:", filename)
        mol_obj_list = [mol for mol in Chem.SDMolSupplier(filename)]
        _, label_data, label_mask = read_label_file(args)  # label name, label, valid/invalid mask of label
    elif args.assay_dir is not None and args.multimodal:
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat = build_all_assay_data(args)
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat = generate_multimodal_data(args, mol_obj_list, label_data, label_mask, dragon_data, task_name_list,mol_id_list, seq, seq_symbol, profeat)
        
    elif args.assay_dir is not None:
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list,mol_id_list, seq, seq_symbol, profeat = build_all_assay_data(args)
        generate_inactive_data(label_data, label_mask)
    elif args.csv_reaxys is not None:
        mol_obj_list, label_data, label_mask, publication_years = parse_csv(args)
    else:
        print("[ERROR] --smarts, --sdf_dir, or --assay_dir is required")
        sys.exit(1)

    return mol_obj_list, label_data, label_mask, dragon_data, task_name_list, seq, seq_symbol, profeat, publication_years


def create_adjancy_matrix(mol):
    mol_adj = Chem.GetAdjacencyMatrix(mol)
    row_num = len(mol_adj)
    adj = np.array(mol_adj, dtype=np.int)
    for i in range(row_num):  # Set diagonal elements to 1, fill others with the adjacency matrix from RDkit
        adj[i][i] = int(1)
    return adj


def create_feature_matrix(mol, args):
    if args.use_sybyl or args.use_gasteiger:
        Chem.SanitizeMol(mol)
    if args.use_gasteiger:
        ComputeGasteigerCharges(mol)
    feature = [atom_features(atom,
        use_sybyl=args.use_sybyl,
        use_electronegativity=args.use_electronegativity,
        use_gasteiger=args.use_gasteiger,
        degree_dim=args.degree_dim) for atom in mol.GetAtoms()]
    if not args.csv_reaxys:
        for _ in range(args.atom_num_limit - len(feature)):
            feature.append(np.zeros(len(feature[0]), dtype=np.int))
    return feature


def write_to_tfrecords(adj, feature, label_data, label_mask, tfrname):
    """
    Writes graph related data to disk.
    """
    adj_row, adj_col = np.nonzero(adj)
    adj_values = adj[adj_row, adj_col]
    adj_elem_len = len(adj_row)
    feature = np.array(feature)
    feature_row, feature_col = np.nonzero(feature)
    feature_values = feature[feature_row, feature_col]
    feature_elem_len = len(feature_row)
    features = Features(
        feature={
            'label': Feature(int64_list=Int64List(value=label_data)),
            'mask_label': Feature(int64_list=Int64List(value=label_mask)),
            'adj_row': Feature(int64_list=Int64List(value=list(adj_row))),
            'adj_column': Feature(int64_list=Int64List(value=list(adj_col))),
            'adj_values': Feature(float_list=FloatList(value=list(adj_values))),
            'adj_elem_len': Feature(int64_list=Int64List(value=[adj_elem_len])),
            'feature_row': Feature(int64_list=Int64List(value=list(feature_row))),
            'feature_column': Feature(int64_list=Int64List(value=list(feature_col))),
            'feature_values': Feature(float_list=FloatList(value=list(feature_values))),
            'feature_elem_len': Feature(int64_list=Int64List(value=[feature_elem_len])),
            'size': Feature(int64_list=Int64List(value=list(feature.shape)))
        }
    )
    ex = Example(features=features)
    with TFRecordWriter(tfrname) as single_writer:
        single_writer.write(ex.SerializeToString())


def main():
    args = get_parser()
    if args.use_deepchem_feature:
        args.degree_dim=11
        args.use_sybyl=False
        args.use_electronegativity=False
        args.use_gasteiger=False

    adj_list = []
    feature_list = []
    label_data_list = []
    label_mask_list = []
    atom_num_list = []
    mol_name_list = []
    seq_symbol_list = None
    dragon_data_list = None
    task_name_list = None
    seq_list = None
    seq = None
    dragon_data = None
    profeat = None
    mol_list=[]
    if args.solubility:
        args.sdf_label="SOL_classification"
        args.sdf_label_active="high"
        args.sdf_label_inactive="low"
    if args.assay_dir is not None:
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list, seq, seq_symbol,profeat, publication_years = extract_mol_info(args)
    else:
        mol_obj_list, label_data, label_mask, _, _, _, _, _, publication_years = extract_mol_info(args)

    if args.vector_modal is not None:
        dragon_data = build_vector_modal(args)
    ## automatically setting atom_num_limit
    if args.atom_num_limit is None:
        args.atom_num_limit=0
        for index, mol in enumerate(mol_obj_list):
            if mol is None:
                continue
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
            if args.atom_num_limit < mol.GetNumAtoms():
                args.atom_num_limit=mol.GetNumAtoms()

    for index, mol in enumerate(mol_obj_list):
        if mol is None:
            continue
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
        # Skip the compound whose total number of atoms is larger than "atom_num_limit"
        if args.atom_num_limit is not None and mol.GetNumAtoms() > args.atom_num_limit:
            continue
        # Get mol. name
        try:
            name = mol.GetProp("_Name")
        except KeyError:
            name = "index_" + str(index)
        mol_list.append(mol)
        mol_name_list.append(name)
        adj = create_adjancy_matrix(mol)
        feature = create_feature_matrix(mol, args)
        if args.csv_reaxys:
            pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
            if publication_years[index] < 2015:
                name += "_train"
            else:
                name += random.choice(["_test", "_eval"])
            tfrname = os.path.join(args.output, str(publication_years[index]), name + '_.tfrecords')
            pathlib.Path(os.path.dirname(tfrname)).mkdir(parents=True, exist_ok=True)
            write_to_tfrecords(adj, feature, label_data[index], label_mask[index], tfrname)
            continue

        atom_num_list.append(mol.GetNumAtoms())
        adj_list.append(dense_to_sparse(adj))
        feature_list.append(feature)
        # Create labels
        if args.sdf_label:
            line = mol.GetProp(args.sdf_label)
            if line.find(args.sdf_label_active) != -1:
                label_data_list.append([0, 1])
                label_mask_list.append([1, 1])
            elif line.find(args.sdf_label_inactive) != -1:
                label_data_list.append([1, 0])
                label_mask_list.append([1, 1])
            else:
                # missing
                print("[WARN] unknown label:",line)
                label_data_list.append([0, 0])
                label_mask_list.append([0, 0])
        else:
            label_data_list.append(label_data[index])
            label_mask_list.append(label_mask[index])
            if dragon_data is not None:
                if dragon_data_list is None:
                    dragon_data_list = []
                dragon_data_list.append(dragon_data[index])
        if args.multimodal:
            if seq is not None:
                if seq_list is None:
                    seq_list, seq_symbol_list = [], []
                seq_list.append(seq[index])
                seq_symbol_list.append(seq[index])
    if args.csv_reaxys:
        sys.exit(0)
    # joblib output
    obj = {"feature": np.asarray(feature_list),
           "adj": np.asarray(adj_list)}
    if not args.sparse_label:
        obj["label"] = np.asarray(label_data_list)
        obj["mask_label"] = np.asarray(label_mask_list)
    else:
        from scipy.sparse import csr_matrix
        label_data = np.asarray(label_data_list)
        label_mask = np.asarray(label_mask_list)
        if args.label_dim is None:
            obj['label_dim'] = label_data.shape[1]
        else:
            obj['label_dim'] = args.label_dim
        obj['label_sparse'] = csr_matrix(label_data.astype(float))
        obj['mask_label_sparse'] = csr_matrix(label_mask.astype(float))
    if task_name_list is not None:
        obj["task_names"] = np.asarray(task_name_list)
    if dragon_data_list is not None:
        obj["dragon"] = np.asarray(dragon_data_list)
    if profeat is not None:
        obj["profeat"] = np.asarray(profeat)
    obj["max_node_num"] = args.atom_num_limit
    mol_info = {"obj_list": mol_list, "name_list": mol_name_list}
    obj["mol_info"] = mol_info
    if args.generate_mfp:
        from rdkit.Chem import AllChem
        mfps=[]
        for mol in mol_list:
            smi=Chem.MolToSmiles(mol)
            mol=Chem.MolFromSmiles(smi)
            mfp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
            mfp_vec=np.array([mfp.GetBit(i) for i in range(2048)],np.int32)
            mfps.append(mfp_vec)
        obj["mfp"] = np.array(mfps)
    ##

    if args.multimodal:
        if seq is not None:
            if args.max_len_seq is not None:
                max_len_seq=args.max_len_seq
            else:
                max_len_seq = max(map(len, seq_list))
            print("max_len_seq:",max_len_seq)
            seq_mat = np.zeros((len(seq_list), max_len_seq), np.int32)
            for i, s in enumerate(seq_list):
                seq_mat[i, 0:len(s)] = s
            obj["sequence"] = seq_mat
            obj["sequence_symbol"] = seq_symbol_list
            obj["sequence_length"] = list(map(len, seq_list))
            obj["sequence_symbol_num"] = int(np.max(seq_mat)+1)

    filename = args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename, compress=3)


if __name__ == "__main__":
    main()
