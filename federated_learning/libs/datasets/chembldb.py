#!/usr/bin/env python
# coding: utf-8
import math
from pathlib import Path
import collections
import sqlite3

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import SaltRemover
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import ClientData

from ..utils import download as _download
from ..utils import (extract_zipfile, create_ids, one_hot, pad_bottom_right_matrix,
                     pad_bottom_matrix, create_mol_feature, check_mol_feature, check_mol_size)


def load_data(FL_FLAG, datapath, max_n_atoms, max_n_types, n_groups,
              subset_ratios: list or int=None):
    """loads the federated tox21 dataset.
    """
    
    if FL_FLAG:
        dataset = create_dataset(datapath, max_n_atoms, max_n_types)
    else:
        dataset = create_dataset(datapath, max_n_atoms, max_n_types)
    return dataset

def _create_mol_feature(mol, max_n_types):
    mol_features = np.array([m.GetAtomicNum() for m in mol.GetAtoms()])
    mol_features = one_hot(mol_features, max_n_types).astype(np.int32)
    return mol_features

def _preprocess_element(data, max_n_atoms, max_n_types, protein_max_seqlen):
    criteria = 10 * math.pow(10, -6) #uM
    chemblvalue = -math.log(criteria, 10)
    smiles = str(data[0])[2:-1] # remove "b''"
    mol = Chem.MolFromSmiles(smiles)
    protein_seq = str(data[1])[2:-1] # remove "b''"
    if True:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.MolFromSmiles(smiles)
    features = pad_bottom_matrix(_create_mol_feature(mol, max_n_types), max_n_atoms).astype(np.float32)
    adjs = pad_bottom_right_matrix(rdmolops.GetAdjacencyMatrix(mol), max_n_atoms).astype(np.float32)
    protein_seq = np.array(_create_one_hot_protein_seq(protein_seq, protein_max_seqlen)).astype(np.float32)
    label = 1 if float(data[-1]) > chemblvalue else 0
    adjs = np.expand_dims(adjs, axis=0)  # for adj channel    
    return adjs, features, protein_seq, label

def _create_one_hot_protein_seq(protein_seq: str, protein_max_seqlen: int):
    one_letter_aa = 'XACDEFGHIKLMNPQRSTVWYOUBJZ'
    vec = []
    for idx in range(protein_max_seqlen):
        if len(protein_seq) > idx:
            c = protein_seq[idx]
            if not c in one_letter_aa:
                print('error!!', c)
            
            vec.append(one_letter_aa.index(c))
        else:
            vec.append(one_letter_aa.index('X'))
    return vec

def _create_element(self, data):
    max_n_atoms = 200
    max_n_types = 150
    adjs, features, protein_seq, label = _preprocess_element(data, max_n_atoms, max_n_types)
    return ({'adjs': adjs,
             'features': features,
             'protein_seq': protein_seq}, label)

def _read_sdf_file(datapath, task_name):
    sdf_name = _task_name_to_sdf_name(task_name)
    file_path = Path(datapath) / sdf_name
    mols = Chem.SDMolSupplier(str(file_path), strictParsing=False)
    data = []
    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            rdmolops.GetAdjacencyMatrix(mol)
        except Exception as e:
            continue
        data.append(mol)
    return data

def create_dataset(datapath, max_n_atoms, max_n_types):
    salt_remover = SaltRemover.SaltRemover()
    criteria = 0
    query = 'select smiles, target_sequence, pchembl_value from activities where valid = 1'
    output_types = (tf.string, tf.string, tf.float64)
    dataset = tf.data.experimental.SqlDataset(
        'sqlite', datapath, query, output_types)
    MAX_N_ATOMS = 200
    MAX_N_TYPES = 110
    PROTEIN_MAX_SEQLEN = 750
    LENGTH_ONE_LETTER_AA = len('XACDEFGHIKLMNPQRSTVWYOUBJZ')

    def map_fn(ele1, ele2, ele3):
        data = (str(ele1), ele2, ele3)
        adjs, features, protein_seq, label = _preprocess_element(data,
                                                                 MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN)
        return adjs, features, protein_seq, label

    def filter_fn(x1, x2, x3):
        smiles = str(x1)[2:-1] # remove "b''"
        mol = Chem.MolFromSmiles(smiles)
        out1 = check_mol_feature(mol, 100)
        out2 = check_mol_size(mol, 150)
        return out1 and out2

    def set_shape(x1, x2, x3, x4):
        data = (str(ele1), ele2, ele3)
        adjs, features, protein_seq, label = _preprocess_element(data,
                                                                 MAX_N_ATOMS, MAX_N_TYPES, PROTEIN_MAX_SEQLEN)
        return adjs, features, protein_seq, label
    
    dataset = dataset.filter(lambda x1, x2, x3: tf.numpy_function(func=filter_fn, inp=[x1, x2, x3],
                                                                  Tout=(tf.bool)))
    dataset = dataset.map(lambda x1, x2, x3: tf.numpy_function(func=map_fn, inp=[x1, x2, x3],
                                                               Tout=(tf.float32, tf.float32, tf.float32, tf.int64)))

    def map_shape_fn(x1, x2, x3, x4):
        x1.set_shape((1, 200, 200))
        x2.set_shape((200, 110))
        x3.set_shape((750,))
        x4.set_shape(())        
        data = collections.OrderedDict({key: x for key, x in zip(['adjs', 'features', 'protein_seq'], (x1, x2, x3))})
        return data, x4
    dataset = dataset.map(lambda x1, x2, x3, x4: map_shape_fn(x1, x2, x3, x4))
    return dataset

