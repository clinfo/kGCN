import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings
from sklearn.utils import class_weight
import joblib
from mendeleev import element
from scipy.sparse import csr_matrix

from kgcn.data_util import dense_to_sparse
from kgcn.preprocessing.utils import read_profeat, read_label_file, parse_csv, create_adjancy_matrix, \
    create_feature_matrix, convert_to_example, save_tfrecords


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
        '--smiles', default=None, type=str,
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
        '--csv_reaxys', default=None, type=str,
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
    parser.add_argument(
        '--tfrecords', action='store_true', default=False,
        help='output .tfrecords files instead of joblib.'
    )
    parser.add_argument(
        '--regression', action='store_true', default=False,
        help='regression'
    )
    return parser.parse_args()


def generate_inactive_data(label_data, label_mask):
    data_index = np.argwhere(label_mask == 1)
    neg_count = 0
    pos_count = 0
    for data_point in data_index:
        if label_data[tuple(data_point)] == 0:
            neg_count += 1
        else:
            pos_count += 1
    print(f"active count: {pos_count}")
    print(f"inactive count: {neg_count}")
    if pos_count > neg_count:
        actives = np.argwhere(label_data == 1)
        np.random.shuffle(actives[:, 1])  # in place
        count = 0
        for inactive_data_point in actives:
            if label_data[tuple(inactive_data_point)] == 0:
                label_mask[tuple(inactive_data_point)] = 1
                count += 1
        print(f"pseudo inactive count: {count}")


def generate_multimodal_data(args, mol_obj_list, label_data, label_mask, dragon_data, task_name_list, mol_id_list, seq,
                             seq_symbol, profeat):
    if not args.no_pseudo_negative:
        enabled_mol_index, enabled_task_index = np.where(label_mask == 1)
        active_count = np.where(label_data[enabled_mol_index, enabled_task_index] == 1)[0].shape[0]
        inactive_count = np.where(label_data[enabled_mol_index, enabled_task_index] == 0)[0].shape[0]
        make_count = active_count - inactive_count
        print(f"[INFO] count = {len(enabled_mol_index)}"
              f"[INFO] active count = {active_count}"
              f"[INFO] inactive count = {inactive_count}"
              f"[INFO] pseudo inactive count = {make_count}"
              f"[INFO] #mols: {len(mol_obj_list)}"
              f"[INFO] #proteins: {len(task_name_list)}")
        if make_count+active_count+inactive_count > len(mol_obj_list)*len(task_name_list):
            print("[WARN] all of the rest data are pseudo negative!")
            # negative_data_index = np.where(label_mask == 0)  #REVIEW why the local variable exists.
            label_mask[label_mask == 0] = 1
        else:
            negative_count = 0
            # negative_data_index = [[], []]
            while negative_count < make_count:
                mol_index = np.random.randint(0, len(mol_id_list), make_count-negative_count)
                protein_index = np.random.randint(0, len(task_name_list), make_count-negative_count)
                flags = label_mask[mol_index, protein_index]
                new_mol_index = mol_index[flags == 0]
                new_protein_index = protein_index[flags == 0]
                if len(new_mol_index) > 0:
                    label_mask[new_mol_index, new_protein_index] = 1
                    label_data[new_mol_index, new_protein_index] = 0  # negative
                    new_index = np.unique(np.array([new_mol_index, new_protein_index]), axis=1)
                    # negative_data_index = np.concatenate([negative_data_index, new_index], axis=1)
                    negative_count += new_index.shape[1]
                    print(f"[INFO] #negative count: {negative_count}")

    x = np.where(label_mask == 1)  # convert_multimodal_label
    # to identify a pair of mol and task
    filename = "multimodal_data_index.csv"
    print(f"[SAVE] mol & task: {filename}")
    with open(filename, "w") as fp:
        for x0, x1 in zip(x[0], x[1]):
            line = f"{str(x0)},{str(x1)}"
            try:
                smi = Chem.MolToSmiles(mol_obj_list[x0])
                t1 = task_name_list[x1]
                line += f",{str(smi)},{str(t1)}"
            except:
                pass
            fp.write(f"{line}\n")
    #
    print(f"[INFO] #data {x[0].shape}")
    ll = label_data[x[0], x[1]]
    dragon_data = dragon_data[x[0]] if dragon_data is not None else None
    mol_obj_list = np.array(mol_obj_list)[x[0]] if mol_obj_list is not None else None
    if seq is not None:
        seq = seq[x[1]]
        seq_symbol = seq_symbol[x[1]]
    profeat = profeat[x[1]] if profeat is not None else None
    max_label = np.max(ll)
    print(f"[INFO] maximum label {max_label}")
    label_dim = int(max_label)+1 if args.label_dim is None else args.label_dim
    print(f"[INFO] label dim {label_dim}")
    if label_dim <= 2:
        new_label_data = np.zeros((ll.shape[0], 2))
        new_label_data[ll == 1, 1] = 1
        new_label_data[ll == 0, 0] = 1
        new_mask_label = np.ones_like(new_label_data)
    else:
        new_label_data = np.zeros((ll.shape[0], label_dim))
        print(ll)
        for i, l in enumerate(ll):
            new_label_data[i, int(l)] = 1
        new_mask_label = np.ones_like(new_label_data)

    return mol_obj_list, new_label_data, new_mask_label, dragon_data, task_name_list, mol_id_list, seq, seq_symbol,\
        profeat


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

        sdf_filename = os.path.join(assay_path, "SDF_wash/SDF_wash.sdf")
        if not os.path.exists(sdf_filename):
            print(f"[PASS] not found: {sdf_filename}")
            return

        dict_id_mol, drop_index = {}, []
        for mol_id_label, mol in zip(df_assay.iterrows(), Chem.ForwardSDMolSupplier(sdf_filename)):
            mol_id = mol_id_label[0]
            if mol is None:
                drop_index.append(mol_id)
                continue
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
        df_dragon = None
        if os.path.exists(dragon_assay_filename1):
            df_dragon = pd.read_csv(dragon_assay_filename1, sep=',', header=None, index_col=0)
        elif os.path.exists(dragon_assay_filename2):
            df_dragon = pd.read_csv(dragon_assay_filename2, sep='\t', header=None, index_col=0)

        # deleting lines in drop_list and converting into dictionary
        self.dragon_data = None
        if df_dragon is not None:
            df_dragon.drop(index=self.drop_index, inplace=True)
            dragon_data = {x[0]: x[1:] for x in df_dragon.itertuples()}
            self.dragon_data = dragon_data

    def _build_seq(self, assay_path):
        seq_filename = os.path.join(assay_path, "protein.fa")
        seq, seq_symbol = None, None
        if os.path.exists(seq_filename):
            seq_symbol = ""
            for line in open(seq_filename):
                seq_symbol += line.strip() if line[0] != ">" else ""  # skip comment line
            seq = [ord(x) - ord("A") for x in seq_symbol]
            seq = list(map(float, seq))
        self.seq = seq
        self.seq_symbol = seq_symbol

    def _build_profeat(self, assay_path, dict_profeat):
        b = os.path.basename(assay_path)
        if dict_profeat is not None:
            self.profeat = dict_profeat[b] if b in dict_profeat else None
            if b not in dict_profeat:
                print(f"[WARN] {b} does not have profeat")

    def build_from_dir(self, args, assay_filename, dict_profeat=None):
        assay_path = os.path.dirname(assay_filename)
        self.path = assay_path
        print(f"[LOAD] {assay_path}")
        self._build_df_assay(args, assay_path)
        self._build_dragon_data(assay_path)
        self._build_seq(assay_path)
        self._build_profeat(assay_path, dict_profeat)
        return self


def concat_assay(assay_list):
    dict_all_assay = None
    dict_all_id_mol = None
    dict_dragon_data = None
    seq, seq_symbol = None, None
    dict_profeat = None

    for assay_data in assay_list:
        if assay_data.dict_assay is not None:
            dict_all_assay = dict_all_assay if dict_all_assay is not None else {}
            dict_all_assay.update({(assay_data.path, str(k)): v for k, v in assay_data.dict_assay.items()})

        if assay_data.dict_id_mol is not None:
            dict_all_id_mol = dict_all_id_mol if dict_all_id_mol is not None else {}
            dict_all_id_mol.update({str(k): v for k, v in assay_data.dict_id_mol.items()})

        if assay_data.dragon_data is not None:
            dict_dragon_data = dict_dragon_data if dict_dragon_data is not None else {}
            dict_dragon_data.update({str(k): v for k, v in assay_data.dragon_data.items()})

        if assay_data.seq is not None and assay_data.seq_symbol is not None:
            seq, seq_symbol = (seq, seq_symbol) if seq is not None else ({}, {})
            seq[assay_data.path] = assay_data.seq
            seq_symbol[assay_data.path] = assay_data.seq_symbol

        if assay_data.profeat is not None:
            dict_profeat = dict_profeat if dict_profeat is not None else {}
            dict_profeat[assay_data.path] = assay_data.profeat
    
    return dict_all_assay, dict_all_id_mol, dict_dragon_data, seq, seq_symbol, dict_profeat


def summarize_assay(args, df_all_assay):
    df_count = df_all_assay.count()
    summary = [df_count]
    count_data = [1, -1]
    column_names = ["count", "count_pos", "count_neg"]
    # + list(map(lambda x: "count_"+str(x), count_data))
    for el in count_data:
        summary.append((df_all_assay == el).sum())
    df_summary = pd.concat(summary, axis=1)
    df_summary.columns = column_names
    summary_filename = f"{os.path.splitext(args.output)[0]}.summary.csv"
    print(f"[SAVE] {summary_filename}")
    df_summary.to_csv(summary_filename)
    return df_summary


def build_all_assay_data(args):
    assay_list = []
    dict_profeat = None
    if args.multimodal:
        dict_profeat = read_profeat()
    for assay_filename in glob.iglob(os.path.join(args.assay_dir, '**/assay.csv'), recursive=True):
        assay_list.append(AssayData().build_from_dir(args, assay_filename, dict_profeat=dict_profeat))

    dict_all_assay, dict_all_id_mol, dict_dragon_data, seq, seq_symbol, dict_profeat = concat_assay(assay_list)
    #
    assay_ids = np.unique([assay_id for assay_id, mol_id in dict_all_assay.keys()])
    mol_ids = np.unique([mol_id for assay_id, mol_id in dict_all_assay.keys()])
    # building label table
    label_data = np.empty((len(mol_ids), len(assay_ids)), dtype=np.float32)
    label_data[:, :] = np.nan
    for i, mi in enumerate(mol_ids):
        for j, aj in enumerate(assay_ids):
            if (aj, mi) in dict_all_assay:
                label_data[i, j] = dict_all_assay[(aj, mi)][0]
    # dropping limitted tasks
    dd = pd.DataFrame(label_data)
    df_summary = summarize_assay(args, dd)
    if args.assay_num_limit is not None:
        names = df_summary.query('count >= '+str(args.assay_num_limit)).index
        label_data = label_data[:, names]
        assay_ids = assay_ids[names]
        # dropping rows consisting of only nan data
        x = ~np.all(np.isnan(label_data), axis=1)
        label_data = label_data[x, :]
        mol_ids = mol_ids[x]
    if args.assay_pos_num_limit is not None:
        names = df_summary.query('count_pos >= '+str(args.assay_pos_num_limit)).index
        label_data = label_data[:, names]
        assay_ids = assay_ids[names]
        # dropping rows consisting of only nan data
        x = ~np.all(np.isnan(label_data), axis=1)
        label_data = label_data[x, :]
        mol_ids = mol_ids[x]
    if args.assay_neg_num_limit is not None:
        names = df_summary.query('count_neg >= '+str(args.assay_neg_num_limit)).index
        label_data = label_data[:, names]
        assay_ids = assay_ids[names]
        # dropping rows consisting of only nan data
        x = ~np.all(np.isnan(label_data), axis=1)
        label_data = label_data[x, :]
        mol_ids = mol_ids[x]
    #
    task_name_list = list(assay_ids)
    mol_id_list = list(mol_ids)
    if dict_all_id_mol:
        mol_list = [dict_all_id_mol[mi] for mi in mol_ids]
    dragon_data = None
    if dict_dragon_data:
        dragon_data = np.array([dict_dragon_data[mi] for mi in mol_ids])
    if seq:
        seq = np.array([seq[mi] for mi in task_name_list])
        seq_symbol = np.array([seq_symbol[mi] for mi in task_name_list])
    profeat_list = None
    if dict_profeat:
        profeat_list = np.array([dict_profeat[mi] for mi in task_name_list])
    #
    mask_label = np.zeros_like(label_data, dtype=np.float32)
    mask_label[~np.isnan(label_data)] = 1
    label_data[np.isnan(label_data)] = 0
    label_data[label_data < 0] = 0
    return mol_list, label_data, mask_label, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat_list


def build_vector_modal(args):
    f = args.vector_modal
    _, ext = os.path.splitext(f)
    df = pd.read_csv(f, sep='\t', index_col=0) if ext == ".des" or ext == ".txt" else pd.read_csv(f)
    return df.values


def extract_mol_info(args):
    dragon_data, task_name_list, seq, seq_symbol, profeat, publication_years = [], [], [], [], [], []
    if args.smarts is not None:
        _, label_data, label_mask = read_label_file(args.label, args.no_header)  # label name, label, valid/invalid mask of label
        with open(args.smarts, "r") as f:
            lines = f.readlines()
        mol_obj_list = [Chem.MolFromSmarts(line) for line in lines]

    elif args.smiles is not None:
        task_name_list, label_data, label_mask = read_label_file(args.label, args.no_header)  # label name, label, valid/invalid mask of label
        with open(args.smiles, "r") as f:
            lines = f.readlines()
        mol_obj_list = [Chem.MolFromSmiles(line) for line in lines]

    elif args.sdf_dir is not None:
        filename = os.path.join(args.sdf_dir, "SDF_wash.sdf")
        if not os.path.exists(filename):
            print(f"[PASS] not found: {filename}")
        mol_obj_list = [mol for mol in Chem.SDMolSupplier(filename)]
        _, label_data, label_mask = read_label_file(args.label, args.no_header)  # label name, label, valid/invalid mask of label

    elif args.sdf is not None:
        filename = args.sdf
        if not os.path.exists(filename):
            print(f"[PASS] not found: {filename}")
        mol_obj_list = [mol for mol in Chem.SDMolSupplier(filename)]
        _, label_data, label_mask = read_label_file(args.label, args.no_header)  # label name, label, valid/invalid mask of label

    elif args.assay_dir is not None and args.multimodal:
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat = \
            build_all_assay_data(args)
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat = \
            generate_multimodal_data(args, mol_obj_list, label_data, label_mask, dragon_data, task_name_list,
                                     mol_id_list, seq, seq_symbol, profeat)
        
    elif args.assay_dir is not None:
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list, mol_id_list, seq, seq_symbol, profeat = \
            build_all_assay_data(args)
        generate_inactive_data(label_data, label_mask)

    elif args.csv_reaxys is not None:
        mol_obj_list, label_data, label_mask, publication_years = parse_csv(args.csv_reaxys)

    else:
        print("[ERROR] --smarts, --sdf_dir, or --assay_dir is required")
        sys.exit(1)

    return mol_obj_list, label_data, label_mask, dragon_data, task_name_list, seq, seq_symbol, profeat,\
        publication_years


def main():
    args = get_parser()
    if args.use_deepchem_feature:
        args.degree_dim = 11
        args.use_sybyl = False
        args.use_electronegativity = False
        args.use_gasteiger = False

    adj_list = []
    feature_list = []
    label_data_list = []
    label_mask_list = []
    atom_num_list = []
    mol_name_list = []
    seq_symbol_list = None
    dragon_data_list = None
    seq_list = None
    seq = None
    dragon_data = None
    profeat = None
    mol_list = []
    train_list = []
    eval_list = []
    test_list = []
    prefix_idx = 0
    if args.solubility:
        args.sdf_label = "SOL_classification"
        args.sdf_label_active = "high"
        args.sdf_label_inactive = "low"
    if args.assay_dir is not None:
        mol_obj_list, label_data, label_mask, dragon_data, task_name_list, seq, seq_symbol, profeat, publication_years =\
            extract_mol_info(args)
    else:
        mol_obj_list, label_data, label_mask, _, task_name_list, _, _, _, publication_years = extract_mol_info(args)

    if args.vector_modal is not None:
        dragon_data = build_vector_modal(args)
    if args.atom_num_limit is None:
        args.atom_num_limit = 0
        for index, mol in enumerate(mol_obj_list):
            if mol is None:
                continue
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
            if args.atom_num_limit < mol.GetNumAtoms():
                args.atom_num_limit = mol.GetNumAtoms()

    if args.use_electronegativity:
        ELECTRONEGATIVITIES = [element(i).electronegativity('pauling') for i in range(1, 100)]
        ELECTRONEGATIVITIES = [e if e is not None else 0 for e in ELECTRONEGATIVITIES]

    for index, mol in enumerate(mol_obj_list):
        if mol is None:
            continue
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
        if args.atom_num_limit is not None and mol.GetNumAtoms() > args.atom_num_limit:
            continue
        try:
            name = mol.GetProp("_Name")
        except KeyError:
            name = "index_" + str(index)
        mol_list.append(mol)
        mol_name_list.append(name)
        adj = create_adjancy_matrix(mol)
        if args.use_electronegativity:
            feature = create_feature_matrix(mol, args.atom_num_limit,
                                            use_electronegativity=args.use_electronegativity,
                                            use_sybyl=args.use_sybyl,
                                            use_gasteiger=args.use_gasteiger,
                                            use_tfrecords=args.tfrecords,
                                            degree_dim=args.degree_dim,
                                            en_list=ELECTRONEGATIVITIES)
        else:
            feature = create_feature_matrix(mol, args.atom_num_limit,
                                            use_electronegativity=args.use_electronegativity,
                                            use_sybyl=args.use_sybyl,
                                            use_gasteiger=args.use_gasteiger,
                                            use_tfrecords=args.tfrecords,
                                            degree_dim=args.degree_dim)

        if args.tfrecords:
            ex = convert_to_example(adj, feature, label_data[index], label_mask[index])
            if args.csv_reaxys:
                if publication_years[index] < 2015:
                    train_list.append(ex)
                else:
                    choice = random.choice(["test", "eval"])
                    if choice == "test":
                        test_list.append(ex)
                    else:
                        eval_list.append(ex)
            if index % 100000 == 0 and index > 0:
                save_tfrecords(args.output, train_list, eval_list, test_list, prefix_idx)
                train_list.clear()
                eval_list.clear()
                test_list.clear()
                prefix_idx += 1
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
                print(f"[WARN] unknown label: {line}")
                label_data_list.append([0, 0])
                label_mask_list.append([0, 0])
        else:
            label_data_list.append(label_data[index])
            label_mask_list.append(label_mask[index])
            if dragon_data is not None:
                dragon_data_list = dragon_data_list if dragon_data_list is not None else []
                dragon_data_list.append(dragon_data[index])
        if args.multimodal:
            if seq is not None:
                seq_list, seq_symbol_list = (seq_list, seq_symbol_list) if seq_list is not None else ([], [])
                seq_list.append(seq[index])
                seq_symbol_list.append(seq[index])
    if args.csv_reaxys:
        save_tfrecords(args.output, train_list, eval_list, test_list, prefix_idx)
    if args.tfrecords:
        with open(os.path.join(args.output, "tasks.txt"), "w") as f:
            f.write("\n".join(task_name_list))
        sys.exit(0)
    # joblib output
    obj = {"feature": np.asarray(feature_list),
           "adj": np.asarray(adj_list)}
    if not args.sparse_label:
        obj["label"] = np.asarray(label_data_list)
        obj["mask_label"] = np.asarray(label_mask_list)
    else:
        label_data = np.asarray(label_data_list)
        label_mask = np.asarray(label_mask_list)
        obj['label_dim'] = label_data.shape[1] if args.label_dim is None else args.label_dim
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
    if not args.regression:
        label_int = np.argmax(label_data_list, axis=1)
        cw = class_weight.compute_class_weight("balanced", np.unique(label_int), label_int)
        obj["class_weight"] = cw

    if args.generate_mfp:
        mfps = []
        for mol in mol_list:
            FastFindRings(mol)
            mfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            mfp_vec = np.array([mfp.GetBit(i) for i in range(2048)], np.int32)
            mfps.append(mfp_vec)
        obj["mfp"] = np.array(mfps)

    if args.multimodal:
        if seq is not None:
            max_len_seq = args.max_len_seq if args.max_len_seq is not None else max(map(len, seq_list))
            print(f"max_len_seq: {max_len_seq}")
            seq_mat = np.zeros((len(seq_list), max_len_seq), np.int32)
            for i, s in enumerate(seq_list):
                seq_mat[i, 0:len(s)] = s
            obj["sequence"] = seq_mat
            obj["sequence_symbol"] = seq_symbol_list
            obj["sequence_length"] = list(map(len, seq_list))
            obj["sequence_symbol_num"] = int(np.max(seq_mat)+1)

    print(f"[SAVE] {args.output}")
    joblib.dump(obj, args.output, compress=3)


if __name__ == "__main__":
    main()
