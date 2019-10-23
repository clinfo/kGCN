## Dataset file
In order to use your own data, you have to create a *dictionary* with the following data format and compress it as a *joblib dump* file.

### Standard GCN

#### *"dense_adj"* (required for GCN)
- a list of adjacency matrices.

#### *"adj"* [optional, alternative to "dense_adj" ] (required for GCN)
- Format: a list of a sparse adjacency matrix.
- A sparse matrix is represented as a tuple ('ind', 'value', 'shape'), where 'ind' expresses the indices of the matrix as a pair of row-col vectors (rows, cols), 'value' is a vector of the entries of the matrix, and 'shape' is a shape of the matrix, that is a pair of the number of rows and the number of cols.

#### *"max_node_num"* (required for GCN)
- Format: a scalar value of the maximum number of nodes in a graph.

#### *"feature"* (required for GCN with feature)
- Format: a list of M by D feature matrices (D is the number of features per node).

#### *"label"* (required for supervised training (of graph-centric GCN))
- Format: a list of E binary label matrices (E is the number of classes).

#### *"node_num"* [optional, node embedding mode]
- Format: a scalar value of the number of all nodes in all graph (= N)

#### *"node"* [optional, node embedding mode]
- Format: a list of a vector for indices of nodes in a graph. (0<= node index < N)

### Multimodal
The following optoins are optional for multimodal mode (e.g. GCN and DNN)

#### *"sequence"*
- Format: a list of symbolic sequences as a integer matrix (the number of graphs x the maximum length of sequences)
- Each element is represented as an integer encoding a symbol (1<= element <=S).

#### *"sequence_length"*
- Format: a list of lengths of sequences. A length of this list should be the number of graphs.

#### *"sequence_symbol_num"*
- Format: a scalar value of the number of symbols in sequences (= S).

#### *"sequence"*
- Format: a list of symbolic sequences as a integer matrix (the number of graphs x the maximum length of sequences)
- Each element is represented as an integer encoding a symbol (1<= element <=S).

#### *"profeat"/"dragon"/"ecfp"*
- Format: a list of vectors as a floating matrix (the number of graphs x the dimension of features)
- "profeat", "dragon", and "ecfp" are processed as the same way. 
