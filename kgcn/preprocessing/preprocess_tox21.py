#!/usr/bin/env python3
# -*- coding: utf-8
"""Preparation tool for tox21 data.
USAGE:
Specify the path to tox21.csv provided by Dr. Kojima.
Channels can be split based on degree. If one adjacency matrix is not split, the approach proposed by Kipf and Welling is used.
A folder named 'tox21_many' or 'tox21_many_split' is generated.
@author: taro.kiritani
"""
import argparse
import os
import pdb
from random import shuffle

import numpy as np
import pandas as pd
from recordclass import recordclass
from rdkit import Chem
import tensorflow as tf

from .preprocessing_utility import get_adj_feature
import tensorflow as tf
if tf.__version__.split(".")[0]=='2':
    from tensorflow.io import TFRecordWriter
else:
    from tensorflow.python_io import TFRecordWriter

def molecule_to_example(molecule, split_adj):
    """
    Create an example for tfrecords from a molecule.
    Args:
        Molecule
        split_adj
    Returns:
        tf example
    """
    adj, feature, _ = get_adj_feature([molecule.mol], normalization=not(split_adj))
    adj_row, adj_col = np.nonzero(adj[0])
    adj_values = adj[0, adj_row, adj_col]
    adj_elem_len = len(adj_row)
    if split_adj:
        degrees = np.sum(adj[0], 0)
        degree_elements = []
        for degree in degrees:
            for d in range(int(degree)):
                degree_elements.append(int(degree) - 1)
    else:
        degree_elements = np.zeros(adj_elem_len).astype(int)
    feature_row, feature_col = np.nonzero(feature[0])
    feature_values = feature[0, feature_row, feature_col]
    feature_elem_len = len(feature_row)
    features = tf.train.Features(
        feature={
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=molecule.label)),
        'mask_label': tf.train.Feature(float_list=tf.train.FloatList(value=molecule.mask_label)),
        'adj_row': tf.train.Feature(int64_list=tf.train.Int64List(value=list(adj_row))),
        'adj_column': tf.train.Feature(int64_list=tf.train.Int64List(value=list(adj_col))),
        'adj_values': tf.train.Feature(float_list=tf.train.FloatList(value=list(adj_values))),
        'adj_elem_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[adj_elem_len])),
        'adj_degrees': tf.train.Feature(int64_list=tf.train.Int64List(value=list(degree_elements))),
        'feature_row': tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature_row))),
        'feature_column': tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature_col))),
        'feature_values': tf.train.Feature(float_list=tf.train.FloatList(value=list(feature_values))), 
        'feature_elem_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_elem_len])),
        'size': tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.shape[1:])))
        }
    )
    ex = tf.train.Example(features=features)
    return ex

def preprocess(dataset, destination_folder, split_adj):
    """
    Preprocesses tox21 data
    Args:
        dataet
        destination_folder
        split_adj
    Return:
        none
    """
    tox21_df = pd.read_csv(dataset)
    task_names = list(tox21_df.columns)
    task_names.remove('smiles')
    task_names.remove('mol_id')
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    train_num = int(len(tox21_df) * 0.8)
    eval_num = int(len(tox21_df) * 0.1)
    test_num = len(tox21_df) - train_num - eval_num
    split = ['train'] * train_num + ['eval'] * eval_num + ['test'] * test_num
    shuffle(split)
    tox21_df['split'] = split

    Molecule = recordclass('Molecule', 'mol label mask_label')
 
    for split in ['train', 'eval', 'test']:
        split_df = tox21_df[tox21_df['split'] == split]
        molecules = []
        for index, row in split_df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            label = (row[:12].to_dense().values == 1).astype(np.float32).tolist()
            mask_label = np.invert(np.isnan(row[:12].values.astype(np.float32))).astype(np.float32)
            molecule = Molecule(mol, label, mask_label)
            molecules.append(molecule)
            with TFRecordWriter(os.path.join(destination_folder, row['mol_id'] + '_' + split  + '.tfrecords')) as single_writer:
                ex = molecule_to_example(molecule, split_adj)
                single_writer.write(ex.SerializeToString())
        #with tf.python_io.TFRecordWriter(os.path.join(dir_name, '_' + split + '.tfrecords')) as dataset_writer:
            #for molecule in molecules:
                #ex = molecule_to_example(molecule, split_adj)
                #dataset_writer.write(ex.SerializeToString())
 
    task_names = "\n".join(task_names)
    with open(os.path.join(destination_folder, 'tasks.txt'), 'w') as text_file:
        text_file.write(task_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='path to tox21.csv')
    parser.add_argument('destination_folder', type=str, help='folder path for tfrecords.')
    parser.add_argument('--split', dest='split', action='store_true')
    parser.add_argument('--no-split', dest='split', action='store_false')
    parser.set_defaults(split=False)
    args = parser.parse_args()
    pdb.set_trace()
    preprocess(args.dataset, args.destination_folder, args.split)
