
# executed in GCN_HOME

fold=$1
name=multitask
python3 gcn.py --config cv/${name}.${fold}.json train > ./cv/log_train${fold}.txt 2>&1
python3 gcn.py --config cv/${name}.${fold}.json infer > ./cv/log_test${fold}.txt 2>&1

