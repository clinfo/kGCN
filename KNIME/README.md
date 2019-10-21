# KNIME-kGCN

KNIME node extension for kGCN.
To use kGCN in KNIME, we provide 
- Virtual box image
- Compiled KNIME modules
- Source codes for developpers

### Example workflow:
- [single-task](testdata/singletask/README.md)  
- [multi-task](testdata/multitask/README.md)  
- [multi-modal](testdata/multimodal/README.md)  


## 1. Virtual box

If Virtual box is installed, you can double-click KNIME-GCN.vbox in this folder

```
User name: kgcn
Password: kgcn
```

After opening the terminal,
you can launch KNIME by the folling command:
```bash
$ ./knime_3.7.2/knime
```

A dialog box about the workspace will be opend, but can be closed.
In this VM image, sample workflows are put in the KNIME explorer
You can easily try the example by opening these workflows:
- singletask_project
- multitask_project
- multimodal_project

Since they have already been executed at the time of loading, if you want to move again, reset and re-execute the workflows.

## 2. Compiled KNIME modules
We also provide compiled knime modules.
First, user should setup the environment for KNIME and kGCN to use these modules.

### Requirements

- kGCN
- kGCN/gcnvisualizer
- Anaconda3
You should create a new Anaconda environment and installing required packages by kGCN.
The default Anaconda environment name is `KGCN`.
```
conda create -n KGCN
```
After that, you should install kGCN by following kGCN README.


### 1. Installation of KNIME-kGCN

1. Installing KNIME from https://www.knime.com/downloads  
2. Copying compiled KNIME modules (jar file) into the `dropins` directory in KNIME.
For example, in Windows, `C:\Program Files\KNIME\dropins` is a default location of the plugin directory.  

### 2. Re-write python-launch scripts
If you change Anaconda environment name from `KGCN`,
you should also change the `kGCN/KNIME/python.sh` (or `kGCN/KNIME/python.bat` in Windows).

### 3. Creating symbolic link for example workflows
To execute example workflows, 
create a symbolic link from the default directory to the your downloaded directory.
```
ln -s <downloaded directory, e.g., ~/kGCN> ~/KNIME_GCN-K
```

### 4. Environment variables

```
export GCNK_SOURCE_PATH=~/kGCN
export GCNK_PYTHON_PATH=~/KNIME_GCN-K/python.sh
export PYTHONPATH=$GCNK_SOURCE_PATH:$PYTHONPATH
```
If you want to use these settings as defaults,
you can add these lines to the end of `~/.bashrc`.

### Notification

If the following error is occurred, 
```
ImportError: Something is wrong with the numpy installation. 
While importing we detected an older version of numpy in ['/home/furukawa/anaconda3/envs/GraphCNN/lib/python3.5/site-packages/numpy']. 
One method of fixing this is to repeatedly uninstall numpy until none is found, then reinstall this version.
```
you should try the following commands:
```bash
$ pip uninstall numpy
$ pip uninstall numpy
$ pip uninstall numpy
$ pip install numpy
```

# For developpers
## Setting up the development environment

We used knime SDK v3.5 since SDK is not distributed from V3.6.
- Downloading KNIME SDK version 3.5.3 for Linux(or KNIME SDK version 3.5.3 for Windows) from 

https://www.knime.com/download-previous-versions  

- Launching KNIME SDK
First, you should specify the repository folder cloned in Workspace by the following operation:

1. Open dialogue:  [File]-[Open Projects from File Ssytem...]  
2. Select  `kGCN\KNIME\GCN-K`  as `Import source`

## Run as debug mode

1. Right-click GCN-K project and select [Run As]-[Eclipse Application]  
2. Select `KNIME` in [Window]-[Perspective]-[Open Perspective]-[Other]  
3. Click OK  

As a result, new nodes are created in `Node Repository`.

## Creating node modules to publish created nodes

1. Right click the GCN-K project and select [Export].
2. Select [Plug-in Development]-[Deployable plug-ins and fragments] and click Next button.
3. Select an output directory and click Finish button

A jar file is generated in the `plugins` directory in the output directory.


## Python scripts for developpers
The KNIME-kGCN modules corresponds to python scripts.
We show the example of python commands called in the each KNIME-kGCN module.

- GraphExtractor
```
prep_adj.py --mol_info <multimodal>/mol_info.jbl --output <multimodal>/adj.jbl
```

- CSVLabelExtractor
```
prep_label.py --label <multimodal>/label.csv --output <multimodal>/label.jbl
```

- SDFReader
```
prep_mol_info.py --sdf <multimodal>/5HT1A_HUMAN.sdf --atom_num_limit 70 --output <multimodal>/mol_info.jbl
```

- AtomFeatureExtractor
```
prep_feat.py --mol_info <multimodal>/mol_info.jbl --output <multimodal>/feat.jbl
```

- AdditionalModalityPreprocessor
```
preprocess_modality.py --profeat <multimodal>/seq_profeat.csv --output <multimodal>/modality1.jbl
```

- AddModality
```
add_modality.py --modality <multimodal>/modality0.jbl --dataset <multimodal>/dataset.jbl --output <multimodal>/dataset1.jbl
```

- GCNScore
```
gcn_score.py --prediction_data <multitask>/prediction.jbl --output <multitask>/result_predict/prediction.csv
```

- GCNScoreViewer
```
gcn_score_viewer.py --prediction_data <multitask>/prediction.jbl --output <multitask>/result_predict/
```

- GCNScoreViewer
```
gcn_score_viewer.py --prediction_data <multitask>/prediction.jbl --output <multitask>/result_predict/ --plot_multitask
```

- GCNGraphViewer
```
graph_viewer.py --directory <multimodal>/visualization
```

- GCNDatasetBuilder
```
prep_dataset.py --label <multimodal>/label.jbl --adjacent <multimodal>/adj.jbl --atom_feature <multimodal>/feat.jbl --output <multimodal>/dataset.jbl
```

- GCNDatasetSplitter
```
split_dataset.py --dataset <multimodal>/dataset1.jbl --output1 <multimodal>/dataset1_split1.jbl --output2 <multimodal>/dataset1_split2.jbl --ratio 0.9
split_dataset.py --dataset <multimodal>/dataset1_split2.jbl --output1 <multimodal>/dataset1_split2_split1.jbl --output2 <multimodal>/dataset1_split2_split2.jbl --ratio 0.9
```
- GCNVisualizer
```
clean_dataset.py --dataset <multimodal>/dataset3_split2_split2.jbl --output <multimodal>/dataset3_split2_split2_clean.jbl
<kgcn>/gcn.py visualize --config <multimodal>/visualize.json
```

- GCNLearner
```
clean_dataset.py --dataset <multimodal>/dataset1_split1.jbl --output <multimodal>/dataset1_split1_clean.jbl
<kgcn>/gcn.py train --config <multimodal>/train.json
```

- GCNPredictor
```
clean_dataset.py --dataset <multimodal>/dataset1_split2.jbl --output <multimodal>/dataset1_split2_clean.jbl
gcn_infer.py infer --config <multimodal>/test.json
```

