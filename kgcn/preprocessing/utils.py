import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from sklearn.preprocessing import LabelEncoder
from tensorflow.python_io import TFRecordWriter
from tensorflow.train import Feature, Features, FloatList, Int64List, Example


def atom_features(atom, en_list=None, explicit_H=False, use_sybyl=False, use_electronegativity=False,
                  use_gasteiger=False, degree_dim=17):
    if use_sybyl:
        import oddt.toolkits.extras.rdkit as ordkit
        atom_type = ordkit._sybyl_atom_type(atom)
        atom_list = ['C.ar', 'C.cat', 'C.1', 'C.2', 'C.3', 'N.ar', 'N.am', 'N.pl3', 'N.1', 'N.2', 'N.3', 'N.4', 'O.co2',
                     'O.2', 'O.3', 'S.O', 'S.o2', 'S.2', 'S.3', 'F', 'Si', 'P', 'P3', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                     'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                     'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    else:
        atom_type = atom.GetSymbol()
        atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                     'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                     'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
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
        results = results + [en_list[atom.GetAtomicNum() - 1]]
    if use_gasteiger:
        gasteiger = atom.GetDoubleProp('_GasteigerCharge')
        if np.isnan(gasteiger) or np.isinf(gasteiger):
            gasteiger = 0  # because the mean is 0
        results = results + [gasteiger]

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)


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


def read_profeat():
    dict_profeat = {}
    filename = "profeat.txt"
    if os.path.exists(filename):
        for line in open(filename):
            arr = line.split("\t")
            dict_profeat[arr[0]] = list(map(float, arr[1:]))
        return dict_profeat
    else:
        return None


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


def parse_csv(args):
    df = pd.read_csv(args.csv_reaxys, dtype={'product': 'str', 'reaction_core': 'str', 'max_publication_year': 'int16'})
    df = df.sample(frac=1, random_state=1234)
    le = LabelEncoder()
    label_data = le.fit_transform(df['reaction_core'])
    with open("class.sma", 'w') as sma:
        sma.write("\n".join(list(le.classes_)))
    label_data = np.expand_dims(label_data, axis=1)
    mol_obj_list = (Chem.MolFromSmarts(product) for product in df['product'])
    label_mask = np.ones_like(label_data)
    publication_years = df['max_publication_year']
    return mol_obj_list, label_data, label_mask, publication_years


def create_adjancy_matrix(mol):
    mol_adj = Chem.GetAdjacencyMatrix(mol)
    row_num = len(mol_adj)
    adj = np.array(mol_adj, dtype=np.int)
    for i in range(row_num):  # Set diagonal elements to 1, fill others with the adjacency matrix from RDkit
        adj[i][i] = int(1)
    return adj


def create_feature_matrix(mol, args, en_list=None):
    if args.use_sybyl or args.use_gasteiger:
        Chem.SanitizeMol(mol)
    if args.use_gasteiger:
        ComputeGasteigerCharges(mol)
    feature = [atom_features(atom,
                             en_list=en_list,
                             use_sybyl=args.use_sybyl,
                             use_electronegativity=args.use_electronegativity,
                             use_gasteiger=args.use_gasteiger,
                             degree_dim=args.degree_dim) for atom in mol.GetAtoms()]
    if not args.tfrecords:
        for _ in range(args.atom_num_limit - len(feature)):
            feature.append(np.zeros(len(feature[0]), dtype=np.int))
    return feature


def convert_to_example(adj, feature, label_data=None, label_mask=None,):
    """
    Writes graph related data to disk.
    """
    adj_row, adj_col = np.nonzero(adj)
    adj_values = adj[adj_row, adj_col]
    adj_elem_len = len(adj_row)
    degrees = np.sum(adj, 0)
    adj_degrees = []
    for ar, ac in zip(adj_row, adj_col):
        if ar == ac:
            adj_degrees.append(0)
        else:
            adj_degrees.append(int(degrees[ar]))
    feature = np.array(feature)
    feature_row, feature_col = np.nonzero(feature)
    feature_values = feature[feature_row, feature_col]
    feature_elem_len = len(feature_row)
    feature = {
            'adj_row': Feature(int64_list=Int64List(value=list(adj_row))),
            'adj_column': Feature(int64_list=Int64List(value=list(adj_col))),
            'adj_values': Feature(float_list=FloatList(value=list(adj_values))),
            'adj_elem_len': Feature(int64_list=Int64List(value=[adj_elem_len])),
            'adj_degrees': Feature(int64_list=Int64List(value=adj_degrees)),
            'feature_row': Feature(int64_list=Int64List(value=list(feature_row))),
            'feature_column': Feature(int64_list=Int64List(value=list(feature_col))),
            'feature_values': Feature(float_list=FloatList(value=list(feature_values))),
            'feature_elem_len': Feature(int64_list=Int64List(value=[feature_elem_len])),
            'size': Feature(int64_list=Int64List(value=list(feature.shape)))
            }
    if label_data is not None:
        label_data = np.nan_to_num(label_data)
        feature['label'] = Feature(int64_list=Int64List(value=label_data.astype(int)))
        feature['mask_label'] = Feature(int64_list=Int64List(value=label_mask.astype(int))),
    features = Features(feature=feature)
    ex = Example(features=features)
    return ex.SerializeToString()


def save_tfrecords(save_dir, train_list, eval_list, test_list, idx):
    with TFRecordWriter(os.path.join(save_dir, f"{idx}_train_.tfrecords")) as writer:
        for e in train_list:
            writer.write(e)
    with TFRecordWriter(os.path.join(save_dir, f"{idx}_test_.tfrecords")) as writer:
        for e in test_list:
            writer.write(e)
    with TFRecordWriter(os.path.join(save_dir, f"{idx}_eval_.tfrecords")) as writer:
        for e in eval_list:
            writer.write(e)
