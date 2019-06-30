# Active Learning

Project for active learning on multitask data, and with molecules. This project is guided by Dr. Ryosuke Kojima at Kyoto University.

## Dependencies
In your conda env, run
```
conda install -c rdkit rdkit
```
. This is necessary because rdkit cannot be installed with pip.

Then, install this package with other dependencies.
```
pip install -e .
```
.

## Usage
```
from active_learning.models import ActiveLearner
from sklearn import svm

# initialize the learner.
learner = ActiveLearner(svm.SCC(probability=True), X_training, Y_training)
# query and sample data points.
query_ids, query_instance = learner.query(X_pool)
learner.teach(query_instance, y_nwe)
```

You can also simulate active querying and sampling if you already have labels for pooled datapoints.

```
from active_learning.models import ActiveLearner
from sklearn import svm

# initialize the learner
learner = ActiveLearner(svm.SCC(probability=True), X_training, Y_training)
query_indices = learner.query_and_teach_n_times(-1, X_pool, Y_pool)
```

The ActiveLearner implements learning on rdkit.Chem.rdchem.Mol objects.

```
learner = ActiveLearner(svm.SCC(probability=True), mols_training, Y_training)
query_ids, qyery_instance = learner.query(mols_pool)
learner.teach(mol_new, y_new)
```

Any estimator that implements fit(), and either predict() or label_distributions_() can be used. Various sampling methods are also supported.

```
from sklearn.semi_supervised import label_propagation

learner = ActiveLearner(label_propagation, mols_training, Y_training, query_strategy='LC') # query_strategy in ['E', 'LC', 'MS', 'RS'].
```
