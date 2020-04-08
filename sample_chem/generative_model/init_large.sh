
python resampling_smiles.py ZINC/6_p0.smi ZINC/6_p0_1000000.smi  --num 1000000
python preprocessing.py ZINC/6_p0_1000000.smi -a 100 --output dataset.single.jbl
python preprocessing.py ZINC/6_p0_1000000.smi -a 100 --multi --output dataset.multi.jbl

