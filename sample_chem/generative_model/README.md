# sample_chem/generative_model

This repository is a example of a graph generative model

Dataset: ZINC

# Building data

## Downloading data

Move to `GraphCNN/sample_chem/generative_model/` and execute the following command

```
sh . /get_dataset.sh
```

The following data will be generated
- ZINC/6_p0.smi

If not possible, download from
https://drive.google.com/drive/folders/15wRLBPHnu6A8emMRM_gU5EQNAQeKk7q8?usp=sharing 

## Resampling of data
Similarly, run the following command in the directory `GraphCNN/sample_chem/generative_model/`.

```
sh . /init.sh
```
This script performs the following operations

- reduce the number of data sets by resampling a portion of the data set, since using all the data would be too much
- create a dataset file (jbl) using preprocessing.py
  - graphs without bond information
  - multi graphs with bond information

The following two files are generated

- dataset.single.jbl
- dataset.multi.jbl

By default, init.sh resamples 10000 data.
If you want to use more data, simply change the part of init.sh that says 10000.

## Learning
To start learning, execute the following commands

In the case of single (without bond information)
```
sh ./run.single.sh
```

In the case of multi (with bond information)
```
sh ./run.multi.sh
```

The detailed settings of the study are described in the following configuration files, respectively.
- config_vae.single.json
- config_vae.multi.json


## Reconstruction

The following two files are generated when training is performed with the above default settings. This is the dataset file with the reconstructed training and validation data.
- recons.train.jbl
- recons.valid.jbl

For the external dataset,
```
kgcn-gen recons --config <config file>
```
will also generate a reconstructed dataset file

### Reconstructed dataset files
The reconstructed dataset file can be read, for example, as follows.

```
import joblib

o=joblib.load("recons.valid.jbl")
print(o.keys())
```

The loaded object is a dictionary with two keys.
- feature' feature matrix for each atom in the molecule
  - number of data x maximum number of atoms in molecule (70) x number of atomic features (75)
- 'dense_adj' The bond probability matrix of the molecule
  - Number of data x type of bond x maximum number of atoms in molecule (70) x maximum number of atoms in molecule (70) 
Note that the bond type is always 1 in single mode is and 5 in multi mode.

An example of a program to reconstruct a molecule from bond types and atomic features is conv_graph.py


For an example of a program that reconstructs molecules from bond types and atomic features, 
please refer to conv_graph.py (see below).

The following 5 dimensions are available for bond types
- Single, Double, Triple, Aromatic, Other

Atomic features have the following 75 dimensions
- 44 dimensions for atom types: 'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl','Yb','Sb','Sn ','Ag','Pd','Co','Se','Ti','Zn','H', # H? 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
- 11 dimensions for GetDegree()
- 7 dimensions for GetImplicitValence()
- 1 dimension for GetFormalCharge()
- 1 dimension for GetNumRadicalElectrons()
- 5 dimensions for GetHybridization(): SP, SP2, SP3, SP3D, SP3D2
- 1 dimension for GetIsAromatic()
- 5 dimensions for GetTotalNumHs()

## Visualization of molecules from a reconstructed dataset file

Specifying the following in conv_graph.py will generate image files visualizing 10 molecules under the directory specified in output_dir.
```
python conv_graph.py recons.valid.jbl --num 10 --output_dir images/ 
```

In multi mode, use the following command
```
python conv_graph.py recons.valid.jbl --num 10 --output_dir images/ --multi
```

If you specify --threshold 0.9 as an option in conv_graph.py, only the bonds with probability greater than or equal to 0.9 will be kept and the numerator will be created.

## Generation

To generate a molecule from scratch instead of rebuilding, execute the following commands
- single mode
```
sh run_gen.single.sh
```
- multi mode
```
sh run_gen.multi.sh
```

Now, we are passing the same dataset as in training, but instead of actually using it in the program, we generate a new dataset,
generate `gen.single.test.jbl` or `gen.multi.test.jbl`.
The format is the same as the reconstructed data set file and can also be visualized as molecules using `conv_graph.py`.
