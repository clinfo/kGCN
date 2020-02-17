mkdir -p result_mm
mkdir -p result_mt
mkdir -p result_st0
mkdir -p result_st1
mkdir -p result_st2
mkdir -p result_st3
# multimodak
kgcn train_cv --config config_mm.json  > log_mm.txt 2>&1 
# multitask
kgcn train_cv --config config_mt.json  > log_mt.txt 2>&1 
# singletask
kgcn train_cv --config config_st0.json --gpu 0 > log_st0.txt 2>&1 &
kgcn train_cv --config config_st1.json --gpu 1 > log_st1.txt 2>&1 &
kgcn train_cv --config config_st2.json --gpu 2 > log_st2.txt 2>&1 &
kgcn train_cv --config config_st3.json --gpu 3 > log_st3.txt 2>&1 &

wait
