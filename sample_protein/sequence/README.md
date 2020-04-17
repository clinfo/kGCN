
# Prediction of protein class from the sequence

## Data preparation

Please prepare "data.tsv" in the following format.
- first column: UniProt ID (string)
- second column: label (integer)

Fasta data can be automatically downloaded using "data.tsv".

```
python 00get_fasta.py
```

Stored sequence data is stored in "protein.fa".

## Data confirmation and modification

```
python 01check.py
```
If you have problems, you can manually modify "protein.fa".

## Construction of dataset

A dataset file "dataset.jbl" is constructed by the following commands
```
python 02make_dataset.py
```

An additional dataset "pos_dataset.jbl" is also constructed, and it contains only positive samples (label=1).

## Training and visualization

```
sh 03run.sh
```

## Summarization of images

```
python 04result_list.py
```

