#!/bin/sh
cd `dirname $0`
cd ../../

kgcn-gen train --config sample_chem/generative_model/config_vae.multi.json --gpu 0

