# Author: yoshi
# Date: 10/02/2019
# Updated: 03/04/2021
# Project: NetworkPrediction
# Script: Split dataset into train/test
# python split_graph.py --input <graph file> --output <output filename> --mode <cv/rate/num>

import sys
import pandas as pd
from sklearn.model_selection import KFold
import argparse

def check_graph(filename):
    print(f'\n== Check input data==\n'
          f'[LOAD] input graph: {filename}')
    graph = pd.read_table(filename, sep='\t', header=None)

    if len(graph.columns) == 3:
        graph.columns = ['nodeA', 'edgetype', 'nodeB']
        node_list = []
        nodepair_list = []
        edgetype_list = []
        for node1, edgetype, node2 in zip(graph['nodeA'], graph['edgetype'], graph['nodeB']):
            nodepair = (node1, node2)
            nodepair = tuple(sorted(nodepair)) 
            nodepair_list.append(nodepair)
            edgetype_list.append(edgetype)
            node_list.append(node1)
            node_list.append(node2)
        nodepair_list_deduplicates = list(set(nodepair_list))
        edgetype_list_deduplicates = list(set(edgetype_list))
        node_list_deduplicates = list(set(node_list))
        print(f'# node: {len(node_list_deduplicates)}\n'
              f'# edge: {len(graph)}\n'
              f'# nodepair: {len(nodepair_list)}\n'
              f'# nodepair post deduplicates: {len(nodepair_list_deduplicates)}\n'
              f'# edgetype: {len(edgetype_list_deduplicates)}')

        if len(nodepair_list) == len(nodepair_list_deduplicates):
            print('[Check]: OK, no edge duplicates. Graph is undirected.')
            graph = graph.sample(frac=1).reset_index(drop=True) # shuffle rows and reset row-index number
            return graph

        else:
            print('[ERROR]: Umm...Exist edge duplicates. Graph seems to be directed.\n')
            sys.exit(1)

    elif len(graph.columns) == 2:
        graph.columns = ['nodeA', 'nodeB']
        nodepair_list = []
        node_list = []
        for node1, node2 in zip(graph['nodeA'], graph['nodeB']):
            nodepair = (node1, node2)
            nodepair = tuple(sorted(nodepair)) 
            nodepair_list.append(nodepair)
            node_list.append(node1)
            node_list.append(node2)
        nodepair_list_deduplicates = list(set(nodepair_list))
        node_list_deduplicates = list(set(node_list))
        print(f'# node: {len(node_list_deduplicates)}\n'
              f'# edge: {len(graph)}\n'
              f'# nodepair: {len(nodepair_list)}\n'
              f'# nodepair post deduplicates: {len(nodepair_list_deduplicates)}')

        if len(nodepair_list) == len(nodepair_list_deduplicates):
            print('[Check]: OK, no edge duplicates. Graph is undirected.')
            graph = graph.sample(frac=1).reset_index(drop=True) # shuffle rows and reset row-index number
            return graph

        else:
            print('[ERROR]: Umm...Exist edge duplicates. Graph seems to be directed.\n')
            sys.exit(1)
 
    else:
        print('[ERROR]: Unknown format.\n')
        sys.exit(1)


def split(graph, output, mode, split_rate, split_num, cv_fold):
    print('\n== Split graph data into train/test ==')
    if mode == 'rate':
        print('[Split mode]: set rate')
        train_graph = graph.sample(frac=split_rate, replace=False, axis=0) #axis=0:row, frac:sampling rate, replace:allow duplicates pick
        train_graph_filename = output + '.train.graph.tsv'
        print(f'[SAVE] train file: {train_graph_filename}\n'
              f'train split rate: {split_rate}\n'
              f'train shape: {train_graph.shape}')
        with open(train_graph_filename, 'w') as f:
            train_graph.to_csv(f, sep='\t', header=False, index=False)
        # Prep test
        test_graph = graph.drop(train_graph.index)
        test_graph_filename = output + '.test.graph.tsv'
        print(f'[SAVE] test file: {test_graph_filename}\n'
               f'test split rate: {1 - split_rate}\n'
              f'test shape: {test_graph.shape}\n')
        with open(test_graph_filename, 'w') as ff:
            test_graph.to_csv(ff, sep='\t', header=False, index=False)

    elif mode == 'num':
        print('[Split mode]: set actual number')
        train_graph = graph.sample(n=split_num, replace=False, axis=0)
        train_graph_filename = output + '.train.graph.tsv'
        print(f'[SAVE] train file: {train_graph_filename}\n'
              f'train shape: {train_graph.shape}')
        with open(train_graph_filename, 'w') as f: 
            train_graph.to_csv(f, sep='\t', header=False, index=False)
        # Prep test
        test_graph = graph.drop(train_graph.index)
        test_graph_filename = output + '.test.graph.tsv'
        print(f'[SAVE] test file: {test_graph_filename}\n'
              f'test shape: {test_graph.shape}\n')
        with open(test_graph_filename, 'w') as ff:
            test_graph.to_csv(ff, sep='\t', header=False, index=False)

    elif mode == 'cv':
        print('[Split mode]: cross validation')
        kf = KFold(n_splits=cv_fold, shuffle=True, random_state=1234) # shuffle data here        
        train_idx_list = []
        test_idx_list = []
        for train, test in kf.split(graph):
            train_idx_list.append(train)
            test_idx_list.append(test)

        for i, (train, test) in enumerate(zip(train_idx_list, test_idx_list)):
            print(f'- generate dataset for cv{i}')
            trainset = graph.iloc[train]
            testset = graph.iloc[test]
            filename_train = output + '_cv' + str(i) + '.train.graph.tsv'
            print(f'[SAVE] train file: {filename_train}\n'
                  f'trainset shape: {trainset.shape}')
            with open(filename_train, 'w') as f:
                trainset.to_csv(f, sep='\t', header=False, index=False)
            filename_test = output + '_cv' + str(i) + '.test.graph.tsv' 
            print(f'[SAVE] test file: {filename_test}\n'
                  f'testset shape: {testset.shape}\n')
            with open(filename_test, 'w') as f:
                testset.to_csv(f, sep='\t', header=False, index=False)

    else:
        print('[ERROR]: you need to select mode.\n')
        sys.exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input graph data')
    parser.add_argument('--output', type=str, help='set output file name')
    parser.add_argument('--mode', type=str, help='select: cv, rate, num')
    parser.add_argument('--cv_fold', type=int, default=5, help='set cv fold to split data')
    parser.add_argument('--split_rate', type=float, default=0.2, help='data split rate for train') 
    parser.add_argument('--split_num', type=int, default=5000, help='data extraction number for train') 
    args = parser.parse_args()

    graph = check_graph(args.input)
    split(graph, args.output, args.mode, args.split_rate, args.split_num, args.cv_fold)

