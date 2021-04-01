#!/bin/sh
cd `dirname $0`

th="15000"
method=$1
echo "method: $method"

echo "\nthreshold: ${th}"
python script/plot_Pk_predgraph.py \
    --graphinput ./result_${method}/score_${method}.txt \
    --traininput ./result_${method}/${method}.train.graph.tsv \
    --th ${th} \
    --train \
    --fig ./result_${method}/predgraph_${method}_train
