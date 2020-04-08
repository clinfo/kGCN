
python resampling_smiles.py ZINC/6_p0.smi ZINC/6_p0_10000.smi  --num 10000
python preprocessing.py ZINC/6_p0_10000.smi -a 70 --output dataset.single.jbl
python preprocessing.py ZINC/6_p0_10000.smi -a 70 --multi --output dataset.multi.jbl

