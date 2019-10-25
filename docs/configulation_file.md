### Configulation file
#### *"model.py"*
model python script

#### *"dataset"*
dataset joblib file

#### *"validation_dataset"*
validation dataset joblib file

#### *"validation_data_rate"*
generating validation dataset by splitting training dataset with *validation_data_rate*

#### *"epoch"*
the maximum numeber of epochs 

#### *"batch_size"*
the number of samples in a minibatch 

#### *"patience"*
patience parameter for early stopping

#### *"learning_rate"*
(initial) learning rate

#### *"shuffle_data"*
shuffling data after loading dataset

#### *"with_feature"*
In GCN, a node has feature or not.

#### *"with_node_embedding"
In GCN, a node has embedding vector or not.
#### *"embedding_dim"*
When `with_node_embedding=True`,
The dimension of an embedding vector.

#### *"normalize_adj_flag"*
enables normalization of adjacency matrices
#### *"split_adj_flag"*
enables splitting adjacency matrices using dgree of a node

#### *"order"*
order of adjacency matrices

#### *"param"*
optional parameters for neural network archtecture
(used in Baysian optimization)

#### *"k-fold_num"*
specifies the number of folds related to train_cv command.

#### *"save_interval"*
inter
#### *"save_model_path"*
path to save model
#### *"save_result_train"*
csv file name to save summarized results (train command)
#### *"save_result_valid"*
csv file name to save summarized results (train command)
#### *"save_result_test"*
csv file name to save summarized results (infer command)
#### *"save_result_cv"*
json file name to save summarized results (train_cv command)

#### *"save_info_train"*
json file name to save detailed information (train command)
#### *"save_info_valid"*
json file name to save detailed information (train command)
#### *"save_info_test"*
json file name to save detailed information (infer command)
#### *"save_info_cv"*
json file name to save cross-validation information (train_cv command)
#### *"make_plot"*
enables plotting results
#### *"plot_path"*
path to save plot data
#### *"plot_multitask"*
plotting results of multitaslk
#### *"profile"*
for profiling using the tensorflow profiler
#### *stratified_kfold*
for using stratified k-fold

