#!/bin/sh
cd `dirname $0`
cd ../../

python gcn_gen.py train --config sample_chem/generative_model/config_vae.single.json --gpu 0

