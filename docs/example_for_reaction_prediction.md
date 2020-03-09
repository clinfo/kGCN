This is a sample usage of a reaction prediction.
- The following additional library is required:
```
pip install mendeleev
```
- First, create the input dataset from a molecule file and a label file
```bash
kgcn-chem -s example_data/mol.sma -l example_data/reaction_label.csv --no_header -o example_jbl/reaction.jbl -a 203 --sparse_label --use_deepchem_feature
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
gcnv -i visualization/mol_0000_task_0_class285_all_scaling.jbl
```
The implementation of extracting reaction template on GitHub at https://github.com/clinfo/extract_reaction_template.git.  
(For instruction of `gcnv`, please see gcnvisualizer/README.md)

#### Reference (Application)

```
@article{Ishida2019,
  author = {Ishida, Shoichi and Terayama, Kei and Kojima, Ryosuke and Takasu, Kiyosei and Okuno, Yasushi},
  title = {Prediction and Interpretable Visualization of Retrosynthetic Reactions Using Graph Convolutional Networks},
  journal = {Journal of Chemical Information and Modeling},
  volume = {59},
  number = {12},
  pages = {5026-5033},
  year = {2019},
  doi = {10.1021/acs.jcim.9b00538},
  URL = { https://doi.org/10.1021/acs.jcim.9b00538 },
}
```