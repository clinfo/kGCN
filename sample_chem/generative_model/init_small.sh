
python resampling_smiles.py ZINC/6_p0.smi ZINC/6_p0_100.smi  --num 100
python preprocessing.py ZINC/6_p0_100.smi -a 70 --output dataset_small.single.jbl
python preprocessing.py ZINC/6_p0_100.smi -a 70 --multi --output dataset_small.multi.jbl

