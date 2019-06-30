import os
import pandas as pd
import argparse
import tensorflow as tf
import numpy as np
from random import seed, shuffle
from rdkit import Chem
from preprocessing_utility import get_adj_feature
import pdb
def preprocessing(dataset, split_adj):
    dir_name = 'rxn'
    if split_adj:
        dir_name = dir_name + '_split'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    num_mols = sum(1 for line in open(dataset)) - 1
    train_num = int(num_mols * 0.8)
    eval_num = int(num_mols * 0.1)
    test_num = num_mols - train_num - eval_num
    split = ['train'] * train_num + ['eval'] * eval_num + ['test'] * test_num
    seed(1122)
    shuffle(split) 
    product_file = dataset.replace('label', 'product')
    labels = open(dataset)
    mols = open(product_file)
    mols.readline() # first line ignored.
    line = ' '
    for i, (label, mol_smarts, sp) in enumerate(zip(labels, mols, split)):
        mol = Chem.MolFromSmarts(mol_smarts)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SANITIZE_ADJUSTHS)
        adj, feature, _ = get_adj_feature([mol], normalization=not(split_adj))
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
        label = label[:-1].split(',')
        label = np.float32(label)
        features = tf.train.Features(
                feature={
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=(label.tolist()))),
                'mask_label': tf.train.Feature(float_list=tf.train.FloatList(value=np.ones_like(label).astype(np.float32))),
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
        with tf.python_io.TFRecordWriter(dir_name + '/'  + sp + str(i) + '.tfrecords') as single_writer:
            single_writer.write(ex.SerializeToString())
    with open(dir_name + '/tasks.txt', 'w') as text_file:
        tasks = ''
        for i in range(1802):
            tasks += str(i) + '\n'
        text_file.write(tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='path to listagg_rxn_mt500_label.csv')
    parser.add_argument('--split', dest='split', action='store_true')
    parser.add_argument('--no-split', dest='split', action='store_false')
    parser.set_defaults(split=False)
    args = parser.parse_args()
    dataset = args.dataset.rstrip('/')
    split = args.split
    preprocessing(dataset, split)
