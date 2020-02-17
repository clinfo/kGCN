#!/bin/sh
cd `dirname $0`

kgcn-chem --assay_dir sample -a 50 --assay_num_limit 100 --sparse_label --output multitask.jbl
kgcn-chem --assay_dir sample -a 50 --assay_num_limit 100 --sparse_label --output multimodal.jbl --multimodal

