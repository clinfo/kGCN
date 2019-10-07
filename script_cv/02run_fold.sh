
# executed in GCN_HOME

fold=$1
name=synth
kgcn train --config test_cv/${name}.${fold}.json 
kgcn infer --config test_cv/${name}.${fold}.json

