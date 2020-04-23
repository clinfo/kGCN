#!/bin/sh
cd `dirname $0`

cd ../../

# run gcn.py train/test
kgcn-gen generate --config sample_chem/generative_model/config_gen.multi.json --gpu 0
#
#kgcn-gen generate --config sample_chem/generative_model/config_gen.multi_only_link.json --gpu 0

#
