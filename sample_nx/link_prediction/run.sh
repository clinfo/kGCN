#!/bin/sh
cd `dirname $0`

python make_data_from_nx.py 
cd ../../
python gcn.py train --config ./sample_nx/link_prediction/config_gcn.json --cpu

