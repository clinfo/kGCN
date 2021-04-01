#!/bin/sh
cd `dirname $0`

method=$1

if [ $method = "ip" ]; then
    mkdir -p result_ip
    echo "\n-train mode"
    kgcn train --config config/config_ip.json --cpu
    echo "\n-infer mode"
    kgcn infer --config config/config_ip.json --cpu

elif [ $method = "distmult" ]; then
    mkdir -p result_distmult
    echo "\n-train mode"
    kgcn train --config config/config_distmult.json --cpu
    echo "\n-infer mode"
    kgcn infer --config config/config_distmult.json --cpu

elif [ $method = "gcn" ]; then
    mkdir -p result_gcn
    echo "\n-train mode"
    kgcn train --config config/config_gcn.json --cpu
    echo "\n-infer mode"
    kgcn infer --config config/config_gcn.json --cpu

else
    echo "[ERROR]: Select methods."
fi

