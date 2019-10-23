
# kGCN: Graph Convolutional Network tools for life science

<img src="https://user-images.githubusercontent.com/1664861/64188778-f9601b00-cead-11e9-8922-da7167fbf9a2.png" height="320px">

## Graph Convolutional Networks

This is a TensorFlow implementation of Graph Convolutional Networks for the task of classification of graphs.

Our implementation of Graph convolutional layers consulted the following paper:

Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

## Installation

A setup script is under construction.
Now, you have to execute the python codes directly.

### Requirements
* python3 (> 3.3)
  * tensorflow (>0.12)
  * joblib

### [Installation (for Ubuntu 18.04)](./docs/installation_for_ubuntu1804.md)

### [Installation (for CentOS 7)](./docs/installation_for_centos7.md)

## Run the demo

For training
```bash
kgcn train --config example_config/sample.json
```
where sample.json is a config file.

For testing and inferrence
```bash
kgcn infer --config example_config/sample.json --model model/model.sample.last.ckpt
```
where model/model.sample.last.ckpt is a trained model file.


## Sample dataset

Our sample dataset file (example.jbl) is created by the following command:

```bash
python example_script/make_example.py
```

When you create your own dataset, you can refer make_sample.py.
This script converts adjacency matrices (example_data/adj.txt), features (example_data/feature.txt), and labels (example_data/label.txt) into the dataset file (example_jbl/sample.jbl)

For example, in training phases, you can specify a dataset as follows:

```bash
kgcn train --config example_config/sample.json --dataset example_jbl/sample.jbl
```

## Configuration

You can specify a configuration file (example_config/sample.json) as follows:

```bash
kgcn train --config example_config/sample.json
```
## The commands of gcn.py

kgcn has three commands: *train*/*infer*/*train_cv*.
You can specify a command as follows:
```bash
kgcn <command> --config example_config/sample.json
```
- *train* command:
The script trains a model and saves it.

- *infer* command:
The script estimates labels of test data using the loaded model.

- *train_cv* command:
The command simplifies cross-validation routines including training stages and estimation(evaluation) stages.
Once you execute this command, cross-validation is performed by running a seriese of training and estimation programs.

### [Configulation file](./docs/configulation_file.md)


## [Dataset file](./docs/dataset_file.md)

In order to use your own data, you have to create a *dictionary* with the following data format and compress it as a *joblib dump* file.

## [Visualization](./docs/visualization.md)

## Directory structure

```
.
├── LICENSE          : LICENSE file
├── README.md        : this file
├── example_config/  : examples of config files
├── example_data/    : examples of adj. files, label files, etc.
├── example_jbl/     : examples of jbl. files
├── example_model/   : examples of model files
├── example_param/   : examples of parameter domain files
├── example_script/ : scripts for the examples
├── gcn.py           : the main engin of this project
├── gcn_gen.py       : an engin for generative models
├── gcn_pair.py      : an engin for ranking models
├── opt_hyperparam.py        :an engin for optimization of hyper parameters
├── kgcn
│   ├── core.py          : a main program files for the GCN model
│   ├── data_util.py     : utilities for data handling
│   ├── error_checker.py : error checker
│   ├── feed.py          : functions to build feed dictionaries
│   ├── feed_index.py    : functions to build feed dictionaries (index base)
│   ├── make_plots.py    : functions to plot graphs
│   ├── layers.py            : GCN-related layers
│   └── visualization.py : functions to visualize trained models
├── graph_kernel/        : graph kernel SVM
├── logs/   : output directory for exmaples
├── model/  : output directory for exmaples
├── result/ : output directory for exmaples
├── visualization/ : output directory for exmaples
├── script                     : utility sctipts
│   ├── make_dataset.py
│   ├── plot_graph.py
│   ├── show_graph.py
│   └── show_label_balance.py
├── script_cv                  : scripts for parallel cross validation
│   ├── 01make_dataset.sh
│   ├── 02run_fold.sh
│   └── make_cross_validation_dataset.py
└── script_synthetic_data    : synthetic data
     ├── synth_generator.py      : random graph
     └── synth_generator_ring.py : random graph with ring

```
## Additional sample1
We provide additional example using synthetic data to discriminate 5-node rings and 6-node rings.
The following command generates synthetic data as text formats:
```bash
python data_generator/synth_generator_ring.py
```

The following command generates .jbl from text:
```bash
python example_script/make_synth.py
```

The following command carries out cross-validation:
```bash
kgcn train_cv --config example_config/synth.json
```
Accuracy and the other scores are stored in:
```
result/synth_cv_result.json
```

More information is stored in:
```
result/synth_info.json
```

## Additional sample2

We prepared additional samples for multimodal and multitask learning.
You can specify a configuration file (sample_multimodal_config.json/sample_multitask_config.json) as follows:

```bash
kgcn --config example_config/multimodal.json train
```
For multimodal, symbolic sequences and graph data are used as the inputs of a neural network.
This configuration file specifies the program of model as "model_multimodal.py", which includes definition of neural networks for graphs, sequences, and combining them.
Please reffer to sample/seq.txt and a coverting program (make_example.py) to prepare sequence data,

```bash
kgcn --config example_config/multitask.json train
```

In this sample, "multitask" means that multiple labels are allowed for one graph.
This configuration file specifies the program of model as "model_multitask.py", which includes definition of a loss function for multiple labels.
Please reffer to sample_data/multi_label.txt and a coverting program (make_sample.py) to prepare multi labeled data,

### Reaction prediction and visualization


This is a sample usage of a reaction prediction.
- The following additional library is required:
```
pip install mendeleev
```
- First, create the input dataset from a molecule file and a label file
```bash
kgcn-chem -s example_data/mol.sma -l example_data/reaction_label.csv --no_header -o example_jbl/reaction.jbl -a 203 --sparse_label
```
- Then, run "gcn.py" by "infer" command to get the accuracy.
```bash
kgcn infer --config example_config/reaction.json
```
This is a sample usage of visualization of the prediction.
- First, install the gcnvisualizer following "kGCN/gcnvisualizer" instruction.
- Then, prepare the input files for gcnvisualizer.
```bash
kgcn visualize --config example_config/reaction.json
```
- Finally, run "gcnv" command to create the figures of the visualization.
```bash
gcnv -i visualization/mol_0000_task_0_active_all_scaling.jbl
```

#### Reference
Shoichi Ishida , Kei Terayama,  Ryosuke Kojima, Kiyosei Takasu, Yasushi Okuno,
[Prediction and Interpretable Visualization of Synthetic Reactions Using Graph Convolutional Networks](http://dx.doi.org/10.26434/chemrxiv.8343995),
ChemRxiv. [DOI: 10.26434/chemrxiv.8343995]

### Generative model

```bash
python gcn_gen.py --config example_config/vae.json train
```

gcn_gen.py is an alternative gcn.py for generative models.
example_config/vae.json is a setting for VAE (Variational Auto-encoder) that is implemented in example_model/model_vae.py

### Hyperparamter optimization

```bash
python opt_hyperparam.py --config ./example_config/opt_param.json  --domain ./example_param/domain.json
```

opt_hyperparam.py is a script for hyperparameter optimization using GPyOpt library (https://github.com/SheffieldML/GPyOpt), a Bayesian optimization libraly.
./example_config/opt_param.json  is a config file to use gcn.py
./example_param/domain.json is a domain file to define hyperparameters and their search spaces.
The format of this file follows "domain" of GPyOpt.
For more information for this json file, see the GPyOpt document(http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/index.ipynb
).

Depending on your environment, it might be necessary to change line 9 (opt_cmd) of opt_hyperparam.py

When you want to change and add hyperparameters, please change domain.json and model file. An example of model file is example_model/opt_param.py in which a hyperparameter is num_gcn_layer.


## License

This edition of kGCN is for evaluation, learning, and non-profit
academic research purposes only, and a license is needed for any other uses.
Please send requests on license or questions to `kojima.ryosuke.8e@kyoto-u.ac.jp`.

