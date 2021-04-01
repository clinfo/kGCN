"""
Author: yoshi, shoichi
Date: 12/19/2019
Updated: 12/19/2020
Project: NetworkPrediction
Description: Script for converting prediction score to table for train/infer mode
Usage: python prediction_score_for_multiprocess_trainfer.py --mode infer --result ./result/test_info.gcn2.jbl --dataset ./dataset.jbl --node ./dataset_node.csv --cutoff 10000 --train --proc_num 2 --output ./result/score_gcn2_cv0.txt --testset ./result/gcn2_cv0.test.graph.tsv --trainset ./result/gcn2_cv0.train.graph.tsv
"""

import argparse
from functools import partial
from multiprocessing import Pool, Manager
import pickle
import pprint
import sys
import time
import joblib
import pandas as pd
from scipy import stats
import numpy as np


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, dict):
        self.__dict__ = dict


def build_node_list(filename):
    """ To convert node ID to gene/chemical name """
    print(f'\n== Prep node list ==\n'
          f'[LOAD]: {filename}')  # node data='dataset_node.csv'

    index2node_mapping = {} # prep dict {'0':'nodeA','1':'nodeB',...}
    with open(filename, 'r') as f:
        for k,l in enumerate(f):
            index2node_mapping[k] = l.strip()
    print(f'#total node: {len(index2node_mapping)}')

    return index2node_mapping


def build_test_label_pairs(filename, mode, cv):
    """ To make test label pair list """
    print(f'\n== Prep test label pairs list ==\n'
          f'[LOAD]: {filename}')
    dataset = joblib.load(filename) # dataset.jbl

    if mode == 'cv':
        test_labels = dataset[cv]['test_labels']

    elif mode == 'infer':
        test_labels = dataset['test_label_list']

    test_label_pairs = []
    for i in test_labels[0]:
        test_label_pair = (i[0], i[2])
        test_label_pair = tuple(sorted(test_label_pair))
        test_label_pairs.append(test_label_pair)

    print(f'#test_label_pairs: {len(test_label_pairs)}\n'
          f'Remove duplicates.')
    test_label_pairs = list(set(test_label_pairs))  # remove duplicates in list of test_label_pairs
    print(f'#test_label_pairs post deduplicates: {len(test_label_pairs)}')

    return test_label_pairs


def build_target_label_pairs(filename, mode):
    """To make all prediction target (train+test) label pair list"""
    # import all edge label data (input data for establish model, train + test) 
    print(f'\n== Prep all target label pairs list ==\n'
          f'[LOAD]: {filename}')
    dataset = joblib.load(filename) # dataset.jbl

    if mode == 'cv':
        label_list = dataset['label_list']

    elif mode == 'infer':
        train_labels = dataset['label_list']
        test_labels = dataset['test_label_list']
        label_list = np.append(train_labels, test_labels, axis=1)

    target_label_pairs = []
    for i in label_list[0]:
        label_pair = (i[0], i[2])
        label_pair = tuple(sorted(label_pair))
        target_label_pairs.append(label_pair)

    print(f'#target_label_pairs: {len(target_label_pairs)}\n'
          f'Remove duplicates.')
    target_label_pairs = list(set(target_label_pairs))  # remove duplicates in list of target_label_pairs
    print(f'#target_label_pairs post deduplicates: {len(target_label_pairs)}')

    return target_label_pairs


def output_test_train(index2node_mapping, test_label_pairs, target_label_pairs):
    """ Prepare test/train label set for export """
    train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs))

    print(f'#target_label_pairs: {len(target_label_pairs)}\n'
          f'#train_label_pairs: {len(train_label_pairs)}\n'
          f'#test_label_pairs: {len(test_label_pairs)}')

    ## prep test
    test_node1_list = []
    test_node2_list = []
    for i in test_label_pairs:
        test_node1 = index2node_mapping[i[0]]
        test_node2 = index2node_mapping[i[1]]
        test_node1_list.append(test_node1)
        test_node2_list.append(test_node2)
    test_table = pd.DataFrame({"node1": test_node1_list, "node2": test_node2_list})

    ## prep train
    train_node1_list = []
    train_node2_list = []
    for k in train_label_pairs:
        train_node1 = index2node_mapping[k[0]]
        train_node2 = index2node_mapping[k[1]]
        train_node1_list.append(train_node1)
        train_node2_list.append(train_node2)
    train_table = pd.DataFrame({"node1": train_node1_list, "node2": train_node2_list})

    return test_table, train_table


def sort_prediction_score(filename, mode, target_label_pairs, test_label_pairs, cutoff, train, index2node_mapping):
    """ Sort prediction result array matrix and Set threshold """
    print('\n== Sort predisction score ==')
    print(f'[LOAD]: {filename}')
    result_data = joblib.load(filename)

    if mode == 'cv':
        print(f'cv fold: {cv}')
        prediction = result_data[cv]['prediction_data']

    elif mode == 'infer':
        prediction = result_data['prediction_data']

    matrix = prediction[0]
    print(f'prediction score matrix: {matrix.shape}')
    # print(f'Prep list of [(score,row,col)] from prediction score results matrix.')
    dim_row = matrix.shape[0]
    dim_col = matrix.shape[1]
    score_row_col = [(matrix[row, col], row, col) for row in range(dim_row) for col in range(row+1, dim_col)]
    print(f'#scores: {len(score_row_col)}')
    totalnode = len(index2node_mapping)
    total_score = int((1+(totalnode-1))*(totalnode-1)/2)
    print(f'#theoretical scores: {total_score}')

    if len(score_row_col) == total_score:
        # sort scores with descending order
        print('Sorting scores with a decending order...')
        score_row_col.sort(reverse=True) # Sort list based on "score" with a decending order

        if train:
            if cutoff == 0:
                score_sort_toplist = score_row_col
                print(f'[Score cutoff] No setting. All scores are used for following steps\n'
                      f'#score post pick score-rank: {len(score_sort_toplist)}\n'
                      f'Completed to prep prediction score-ordered list including train labels.')
                return score_sort_toplist

            else:
                score_sort_toplist = score_row_col[:cutoff]  # args.cutoff: Select top score ranking to export
                print(f'[Score cutoff]: {cutoff}\n'
                      f'#score post pick score-rank: {len(score_sort_toplist)}\n'
                      f'Completed to prep prediction score-ordered list including train labels.')
                return score_sort_toplist

        else:
            print('[ERROR] train set are NOT included in the sorted prediction score list.')
            sys.exit(1)

            #print(f'(Train labels are excluded for preparing score-ordred list.)\n'
            #      f'Pick toplist by score_rank.')
            train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs)) # Prep target,test,train label list
            score_row_col_ = [i for i in score_row_col if (i[1], i[2]) not in set(train_label_pairs)]
            score_row_col_.sort(reverse=True)
 
            if cutoff == 0:
                score_sort_toplist = score_row_col
                return score_sort_toplist
            else:
                score_sort_toplist = score_row_col_[:cutoff]
                print(f'#score post pick score-rank: {len(score_sort_toplist)}\n'
                      f'Completed to prep prediction score-ordered list w/o train labels.')
                return score_sort_toplist

    else:
        print('[ERROR] the number of adopted prediction score is NOT same as the theoretical value.')
        sys.exit(1)


def convert(score_sort_toplist, target_label_pairs, test_label_pairs, index2node_mapping, train, total_list):
    """
    Preprare for Conversion score-sorted list [(score,row,col),...] into table.
    total_list = (scores, rows, cols, gene1, gene2, train_edge, test_edge, new_edge)
    """
    tmp_list = []
    if train:
        for i in score_sort_toplist:
            scores = i[0]
            row = i[1]
            gene1 = index2node_mapping[row]
            col = i[2]
            gene2 = index2node_mapping[col]
            prediction_label_pair = (row, col)
            if prediction_label_pair in target_label_pairs:
                if prediction_label_pair in test_label_pairs:
                    tmp_list.append([scores, row, col, gene1, gene2, 0, 1, 0])
                else:
                    tmp_list.append([scores, row, col, gene1, gene2, 1, 0, 0])
            else:
                tmp_list.append([scores, row, col, gene1, gene2, 0, 0, 1])

    else:
        for i in score_sort_toplist:
            scores = i[0]
            row = i[1]
            gene1 = index2node_mapping[row]
            col = i[2]
            gene2 = index2node_mapping[col]
            prediction_label_pair = (row, col)
            if prediction_label_pair in test_label_pairs:
                tmp_list.append([scores, row, col, gene1, gene2, 0, 1, 0])
            else:
                tmp_list.append([scores, row, col, gene1, gene2, 0, 0, 1])
    total_list.extend(tmp_list)


def process_table(rows, cols, gene1, gene2, scores, train_edge, test_edge, new_edge):
    """ To build a table """
    print('\n== Process curated prediction score to build a table ==')
    table = pd.DataFrame({
        "row": rows,
        "col": cols,
        "gene1": gene1,
        "gene2": gene2,
        "score": scores,
        "train_edge": train_edge,
        "test_edge": test_edge,
        "new_edge": new_edge
    })
    # print('#table shape: ', table.shape)
    table = table.assign(score_ranking=len(table.score) - stats.rankdata(table.score, method='max') + 1)
    print('Sort the table with score-descending order.')
    table_sort_score = table.sort_values(by='score', ascending=False)
    table_sort_score = table_sort_score[['row', 'col', 'gene1', 'gene2', 'score',
                                         'score_ranking', 'train_edge', 'test_edge', 'new_edge']]
    print(f'#final table shape: {table.shape}')
    return table_sort_score


def enrichment(target_label_pairs, test_label_pairs, table_sort_score, train, index2node_mapping):
    print('\n== Calculate enrichment ==')
    train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs)) # prep train edges list
    totalnode = len(index2node_mapping)
    total = int((1+(totalnode-1))*(totalnode-1)/2)
    total_wo_train = total - len(train_label_pairs)  # remove train edges from total
    total_test_edges = len(test_label_pairs)
    table_wo_train = table_sort_score[table_sort_score.train_edge == 0]  # prep table w/o train edges (remove train from the table)
    print(f'Summary of edges attribution\n'
          f'#total as scored: {total}\n'
          f'#total_w/o_train_edges: {total_wo_train}\n'
          f'#total_target_edges: {len(target_label_pairs)}\n'
          f'#total_train_edges: {len(train_label_pairs)}\n'
          f'#total_test_edges: {len(test_label_pairs)}\n')

    # enrichment calcucation
    top = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Set top%: 1%, 3%, 5%
    for i in top:
        ratio = i*0.01
        top_ratio = round(total_wo_train*ratio)  # calculate the number of top list based on top%
        table_wo_train_toplist = table_wo_train.iloc[:top_ratio, ]  # pick top list from the table w/o train edges
        test_edges_in_toplist = len(table_wo_train_toplist[table_wo_train_toplist.test_edge == 1].index)
        test_edges_enrichment = test_edges_in_toplist/total_test_edges
        print(f'#top%: {i}\n'
              f'#top_ratio: {top_ratio}\n'
              f'#test_edges_in_toplist: {test_edges_in_toplist}\n'
              f'#test edges enrichment top{i}%: {test_edges_enrichment}\n')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, help="input result: gcn_cv.jbl")
    parser.add_argument('--dataset', type=str, help="input dataset: dataset.jbl")
    parser.add_argument('--node', type=str, help="input dataset node: dataset_node.csv")
    parser.add_argument('--output', type=str, help="output: score.txt")
    parser.add_argument('--testset', type=str, help="output test label: test_gcn_cv.graph.tsv")
    parser.add_argument('--trainset', type=str, help="output train label: train_gcn_cv.graph.tsv")
    parser.add_argument('--cutoff', default=10000, type=int, help='pick score ranking from 1 to score_rank. If set 0, all socres are used.')
    parser.add_argument('--train', action="store_true", help="default: exclude train label at score ranking list")
    parser.add_argument('--proc_num', type=int, default=1, help="a number of processors for multiprocessing.")
    parser.add_argument('--mode', type=str, help="cv or infer")
    parser.add_argument('--cv', default=0, type=int, help="cross validation: select 0,1,2,3,4")
    args = parser.parse_args()
    print('\n== args summary ==')
    pprint.pprint(vars(args))
    return args


def split_list(l, n):
    return [l[i::n] for i in range(n)]


def main():
    args = get_parser()
    start_time = time.time()

    index2node_mapping = build_node_list(args.node)
    test_label_pairs = build_test_label_pairs(args.dataset, args.mode, args.cv)
    target_label_pairs = build_target_label_pairs(args.dataset, args.mode)
    score_sort_toplist = sort_prediction_score(args.result, args.mode, target_label_pairs, test_label_pairs,
                                               args.cutoff, args.train, index2node_mapping)

    print('\n== Start convesion of prediction scores ==')
    print(f'Train labels are {["included" if args.train else "excluded"][0]}.')
    print('Processing...')
    n_proc = args.proc_num
    pool = Pool(processes=n_proc)
    split_score_sort_toplist = split_list(score_sort_toplist, n_proc)
    with Manager() as manager:
        total_list = manager.list()
        convert_ = partial(convert, target_label_pairs=set(target_label_pairs), test_label_pairs=set(test_label_pairs),
                           index2node_mapping=index2node_mapping, train=args.train, total_list=total_list)
        pool.map(convert_, split_score_sort_toplist)
        scores = [l[0] for l in total_list]
        rows = [l[1] for l in total_list]
        cols = [l[2] for l in total_list]
        gene1 = [l[3] for l in total_list]
        gene2 = [l[4] for l in total_list]
        train_edge = [l[5] for l in total_list]
        test_edge = [l[6] for l in total_list]
        new_edge = [l[7] for l in total_list]
        print('Completed conversion.')

    table_sort_score = process_table(rows, cols, gene1, gene2, scores, train_edge, test_edge, new_edge)
    enrichment(target_label_pairs, test_label_pairs, table_sort_score, args.train, index2node_mapping)

    print(f'\n== Export the processed prediction score result ==\n'
          f'[SAVE] score file: {args.output}')
    with open(args.output, 'w') as f:
        table_sort_score.to_csv(f, sep='\t', header=True, index=False)

    print(f'\n== Export the test/train label set ==\n'
          f'[SAVE] testset file: {args.testset}\n'
          f'[SAVE] trainset file: {args.trainset}')
    test_table, train_table = output_test_train(index2node_mapping, test_label_pairs, target_label_pairs)
    with open(args.testset, 'w') as f:
        test_table.to_csv(f, sep='\t', header=False, index=False)
    with open(args.trainset, 'w') as f:
        train_table.to_csv(f, sep='\t', header=False, index=False)

    elapsed_time = time.time() - start_time
    print(f'\n#time: {elapsed_time} sec\n'
          f'-- fin --\n')


if __name__ == '__main__':
    main()

