
# Protein interaction 

This is an example of a problem that predicts the assay activities of a protein and a ligand. 
This example contains three types of graph convolutional neural networks: single task, multi task, multimodal neural networks.
The single task neural network predicts the activity related to a protein from a chemical structre represented as a graph.
The multi task neural network predicts the activities related to the all proteins from a chemical structre.
The multi task neural networks are usually known to perform better than single tasks.
Although these networks does not use the information related proteins,
multimodal neural networks predict the activity from a protein sequence and chemical structure in this example.

Predicting compound-protein interactions (CPIs) has played an important role in drug discovery [1], CPIs prediction methods using deep learning have achieved excellent performances [2,3].

```
[1] Michael, J. K., Vincent, S., Predicting new molecular targets for known drugs, Nature, 462:175-181, 2009.
[2] Tsubaki, M., Tomii, K., Sese, J., Compound-Protein interaction prediction with end-to-end learning of neural networks for graphs and sequences, Bioinformatics, 35:309-318, 2018.
[3] Hamanaka, M., Taneishi, K., Iwata, H., Ye, J., Pei, J., Hou, J., and Okuno, Y., CGBVS-DNN: Prediction of Compound‐protein Interactions Based on Deep Learning, Molecular Informatics, 2016.
```

# Download dataset
To evaluate our framework, we prepare a preprocessed dataset from ChEMBL ver.20 Database.
We define threshold for active/inactive as 30uM.
The dataset can be downloaded from the following link:


This dataset contains four-type assays: MMP-3, MMP-9, MMP-12, and MMP-13.
Each assay conatains data sample (Compound) as 
- MMP-3: 2095 compounds
- MMP-9: 2829 compounds
- MMP-12: 533 compounds
- MMP-13:2607 compounds

* Dataset: PubChem

The kGCN supports many types of data descripter for a ligand and a protein.
For example, kGCN supports graph representatinon for GCN and vector representation, such as ECFP and DRAGON, for standard neural networks.
Also, to represent a protein, kGCN supports an amino-acid sequence and vector representation such as PROFEAT descriptors.
This example uses graph representation for a compound and sequence representation for a sequence.


# Conversion of dataset

To simplify the experiment, we remove the molecules with more than 50 atoms.
Since the data set was unbalanced, negative examples are generated.
Althogh there exists various ways to generate negative examples,
now we have simply created compound-protein pairs at random from the non-positive pairs so that the negative examples are the same number as the positive examples.
Finally, we obtain compound-protein pairs by the following commands:

```
sh build_dataset.sh
```
This script generates two joblib files for multi-task and multi-modal.

- #data    : 9503
- #features: 81
- The maximum number of atoms in a molecule: 50


Datasets for the single-task GCN can be easily generated from the multitask dataset by the following script:
```
python convert_multitask2singletask.py
```

This script generate four dataset files corresponded to the tasks:
- singletask.0.jbl 
- singletask.1.jbl 
- singletask.2.jbl 
- singletask.3.jbl 


# Cross-validation
kGCN supports two types of commands for cross-validation: full-auto command and parallel execution.

## Full-auto command for cross-validation

To execute full-auto command for cross-validation,
simply executes the following command:

```
kgcn train_cv --config <config file>
```

In this example, we prepare the following config files:
- config_st0.json: configuration of single-task GCN for the 1st task
- config_st1.json:  configuration of single-task GCN for the 2nd task
- config_st2.json:  configuration of single-task GCN for the 3rd task
- config_st3.json:  configuration of single-task GCN for the 4th task
- config_mt.json:  configuration of multi-task GCN
- config_mm.json: configuration of multi-modal GCN


## Parallel execution for cross-validation
The full-auto command described above is executed with the single thread.
kGCN also supports parallel execution.

The whole script is written in `run_para_cv.sh`

```
sh run_para_cv.sh
```

The follwing description explains the contents of this script.

### splitting a dataset into training and test

kGCN supports utility command to separate the dataset into training and test datasets.

```
kgcn-cv-splitter --config <original config file> --cv <path>
```
where the cv option specifies the directories for the separated datasets and auxiliary files.
This command also separates configuration into the separated configuration files, which are generated in the <path> directory.

### training and test
By using these generated configuration files, training and test commands can be executed without interfering with each other even if parallel execution is performed.
Therefore, each fold in the cross-validation can be executed in parallel.
The training and test command is the same with the original command except for using the generated configuration file.
```
kgcn train --config <generated config file>
kgcn predict --config <generated config file>
```

### Visualization

```
kgcn visualize --config <generated config file> --visualization_header fold_N
```
The header of the output file names can be specified by the  --visualization_header  option.
This option is used to organize output files.

This command generates visualization data file.
To convert this file into picture files, the following command should be carried out:
```
gcnv -i <generated .jbl file>
```
For example, the command `gcnv -i ./viz_st0/fold0_0003_task_0_active_all_scaling.jbl` outputs 
`fold2_0518_task_0_inactive_all_scaling_mol.svg` to visualize atom importaces in a molecule related to single-task data 0 in fold-0.
For more details, please see the `gcnv` README in `kGCN/gcnvisualizer`.

# 81-dimensional atom features
kGCN uses the following features by default.
kGCN also provides more features like SYBIL atom types, gasteiger charge, and electron negativity and user can easily try these features.


- 44 dimensions for atom types: 'C','N','O','S','F', 'Si','P','Cl','Br','Mg', 'Na','Ca','Fe','As','Al', 'I','B','V','K','Tl', 'Yb','Sb','Sn','Ag','Pd', 'Co','Se','Ti','Zn','H', # H? 'Li','Ge','Cu','Au','Ni', 'Cd','In','Mn','Zr','Cr', 'Pt','Hg','Pb','Unknown'
- 17 dimensions for GetDegree()
- 7 dimensions for GetImplicitValence()
- 1 dimension for GetFormalCharge()
- 1 dimension for GetNumRadicalElectrons()
- 5 dimensions for GetHybridization(): SP, SP2, SP3, SP3D, SP3D2
- 1 dimension for GetIsAromatic()
- 5 dimension for GetTotalNumHs()

