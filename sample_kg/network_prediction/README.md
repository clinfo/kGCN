# Network Prediction

### Requirements
* scipy
* networkx (2.2)
* powerlaw
* thinkx
* multiprocessing


### Directory structure
```
.
├── script/
│     └── scripts for preprocessing and analysis
├── data/
│     └── dataset directory
├── config/
│     └── config files
└── model_py/
      └── model files
```


## Example Usage

### 1. Make dataset
Any triplet graph file is acceptable.  
If you use the model network (Erdős-Rényi model / Watts–Strogatz model / Barabási–Albert model), run the code below prior to preprocessing.  
(The model network datasets used in the paper are placed in `./data/`)


**Generate the model network**
```bash
python ./script/modelgraph_generation.py --graph <er/ws/ba> --output ./data/dataset.graph.tsv --fig ./data/sample
```

**Preprocessing**
```bash
python ./script/preprocessing_graph_trainfer.py --input ./data/dataset.graph.tsv --output ./data/dataset --mode num
```

```bash
sh make_dataset.sh
```

 - `dataset.jbl` : input data for step 2.
 - `dataset_node.csv` : node list file. The i-th row describes i-th node name. This file is required for the following analysis.


### 2. Run GCN/DistMult/IP
**Set config files**
 - `./config/config_<gcn/distmult/ip>.json`

**Start train and infer**
```bash
sh run.sh <gcn/distmult/ip>
```
- `pred_data.jbl` : prediction data file.

### 3. Calculate enrichment and process the prediction data

```bash
sh run_enrichment.sh <gcn/distmult/ip>
```

arguments;
 - result : prediction data file generated through GCN/DistMult/IP (`pred_data.jbl`). (step 2)
 - dataset : input dataset file (`dataset.jbl`). (step 1)
 - node : node list file (`dataset_node.csv`). (step 1)
 - cutoff : score threshold.
 - proc_num : the number of CPU to use.
 - output : main output file (`score_<gcn/distmult/ip>.txt`).
 - testset : test graph output file (`<gcn/distmult/ip>.test.graph.tsv`).
 - trainset: train graph output file (`<gcn/distmult/ip>.train.graph.tsv`).


### 4. Construct and analyze a predicted network

This code constructs a predicted network, calculates network properties, and generates a degree distribution figure.

```bash
sh run_plot_Pk_predgraph.sh <gcn/distmult/ip>
```

arguments;
 - graphinput : predicted graph file (`score_<gcn/distmult/ip>.txt`). (step 3)
 - traininput : train graph output file (`<gcn/distmult/ip>.train.graph.tsv`). (step 3)
 - th : arbitrary link threshold for determining the number of links in the predicted network.

## Reference
```
@article{tanaka2021,
  title={Complex network prediction using deep learning}, 
  author={Yoshihisa Tanaka and Ryosuke Kojima and Shoichi Ishida and Fumiyoshi Yamashita and Yasushi Okuno},
  year={2021},
  eprint={2104.03871},
  archivePrefix={arXiv},
  primaryClass={physics.soc-ph}
}
```
