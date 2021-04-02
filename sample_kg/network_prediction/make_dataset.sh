#!/bin/sh
cd `dirname $0`

python script/preprocessing_link_pred.py \
    --train data/dataset.train.graph.tsv \
    --test data/dataset.test.graph.tsv \
    --output_jbl data/dataset.jbl \
    --output_csv data/dataset_node.csv

