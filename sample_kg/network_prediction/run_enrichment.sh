#!/bin/sh
cd `dirname $0`

method=$1
echo "method: $method"

python script/predscore.py \
    --result ./result_${method}/pred_data.jbl \
    --dataset ./data/dataset.jbl \
    --node ./data/dataset_node.csv \
    --cutoff 1500000 \
    --train \
    --proc_num 2 \
    --mode infer \
    --output ./result_${method}/score_${method}.txt \
    --testset ./result_${method}/${method}.test.graph.tsv \
    --trainset ./result_${method}/${method}.train.graph.tsv

