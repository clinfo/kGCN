"""Script that suggests the next data points based on existing
data points.
"""
import copy
import logging
import pdb

import numpy as np
from rdkit.Chem import Mol, AllChem
from scipy import stats
from sklearn.semi_supervised import label_propagation, LabelPropagation, LabelSpreading
from sklearn import svm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


def query_next_data_points(X: np.array,
                           Y: np.array,
                           label_presence=None,
                           algorithm='LP',
                           kernel='rbf',
                           gamma=20,
                           n_neighbors=7,
                           max_iter=1000,
                           tol=0.001,
                           n_jobs=None,
                           alpha=0.2,
                           US_strategy='E'):
    """

    Suggests the next data points for sampling.
    Args:
        X: n by d numpy array where n is the number of data points and d is
        the dimension of each data point.
        Y: the labels for X. 1d array or n by t numpy array where t is the number of tasks
        for multitasking. -1 means missing data points.
        label_presence: a list-like object of boolean that tells which data points luck labels.
        If None, unlabeled data points will be inferred from Y.
        algorithm: classifier used for label propagation. One of ['LP',
        'LS', 'SVM'] for sklearn.semi_supervised.LabelPropagation,
        sklearn.semi_supervised.LabelSpreading, or sklearn.svm.SVC, respectively.
        kernel: the kernel used in algorithm. See options in sklearn doc.
        gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        n_neighbors: Parameter for knn kernel.
        max_iter: maximum number of iterations allowed.
        tol: convergence tolerance.
        n_jobs: the number of parallel jobs to run.
        US_strategy: uncertainty sampling strategy. One of ['MS', 'LC', 'E', 'RS'].
    Returns:
        The index of next data points in X and Y suggested by the algorithm.

    """
    _Y = Y.copy()
    if len(_Y.shape) == 1:
        _Y = np.expand_dims(_Y, 1)
    task_num = _Y.shape[1]
    if label_presence is None:
        unlabeled_indices = (np.ones(task_num, dtype=int) * -1 == _Y).all(1)
        unlabeled_indices = np.nonzero(unlabeled_indices)[0]
    else:
        unlabeled_indices = np.nonzero(np.logical_not(label_presence))[0]
        _Y[unlabeled_indices] = np.ones([len(unlabeled_indices), task_num
                                         ]) * -1

    labeled_indices = set(range(_Y.shape[0])) - set(unlabeled_indices)
    labeled_indices = sorted(labeled_indices)

    predicted_labels = []
    predicted_all_labels = []
    label_distributions = []
    label_distributions_all = []
    classes = []

    if US_strategy == 'RS':
        index_most_uncertain = np.random.permutation(unlabeled_indices)[:5]
        u_score_list = [0.5 for i in range(len(label_distributions))]
        return index_most_uncertain

    if algorithm in ['LS', 'LP']:
        if algorithm == 'LS':
            lp_models = [
                label_propagation.LabelSpreading(kernel, gamma, n_neighbors,
                                                 alpha, max_iter, tol, n_jobs)
                for _ in range(task_num)
            ]
        else:
            if n_jobs is None:
                nj = 1
            else:
                nj = n_jobs
            lp_models = [
                label_propagation.LabelPropagation(
                    kernel=kernel,
                    gamma=gamma,
                    n_neighbors=n_neighbors,
                    max_iter=max_iter,
                    tol=tol,
                    n_jobs=nj) for _ in range(task_num)
            ]
        for task_index, lp_model in enumerate(lp_models):
            lp_model.fit(X, _Y[:, task_index])
            predicted_labels.append(lp_model.transduction_[unlabeled_indices])
            predicted_all_labels.append(lp_model.transduction_)
            label_distributions.append(
                lp_model.label_distributions_[unlabeled_indices])
            label_distributions_all.append(lp_model.label_distributions_)
            classes.append(lp_model.classes_)

    elif algorithm == 'SVM':
        lp_models = [
            svm.SVC(probability=True, C=10, gamma=gamma)
            for _ in range(task_num)
        ]
        # train SVM
        for task_index, lp_model in enumerate(lp_models):
            y = _Y[:, task_index]
            y_labeled = y[y != -1]
            x = X[y != -1]
            lp_model = lp_models[task_index]
            lp_model.fit(x, y_labeled)
            predicted_labels.append(lp_model.predict(X[unlabeled_indices]))
            predicted_all_labels.append(lp_model.predict(X))
            label_distributions.append(
                lp_model.predict_proba(X[unlabeled_indices]))
            label_distributions_all.append(lp_model.predict_proba(X))
            classes.append(lp_model.classes_)

    # select up to 5 examples that the classifier is most uncertain about
    if US_strategy == 'E':
        pred_entropies_list = []
        for task_index in range(_Y.shape[1]):
            entropies = stats.distributions.entropy(
                label_distributions[task_index].T)
            pred_entropies_list.append(entropies)
        pred_entropies = np.vstack(pred_entropies_list, )
        pred_entropies = np.sum(pred_entropies, axis=0)
        uncertainty_score_list = pred_entropies / np.max(pred_entropies)
        sorted_entropies = np.argsort(pred_entropies)
        index_most_uncertain = unlabeled_indices[sorted_entropies[-5:]]

    elif US_strategy == 'LC':
        label_distributions = np.stack(label_distributions).mean(axis=0)
        u_score_list = 1 - np.max(
            label_distributions, axis=1)  # could use just np.min, but..
        #for i in range(len(label_distributions_all)):
        #    print(i, label_distributions_all[i], grid_list_std[i])
        #print('label_distributions_all', label_distributions_all)
        #print('u_score_list', u_score_list)
        sorted_score = np.argsort(u_score_list)
        index_most_uncertain = unlabeled_indices[sorted_score[-5:]]

    elif US_strategy == 'MS':
        label_distributions = np.stack(label_distributions).mean(axis=0)
        u_score_list = []
        for pro_dist in label_distributions:
            pro_ordered = np.sort(pro_dist)[::-1]
            margin = pro_ordered[0] - pro_ordered[1]
            u_score_list.append(margin)

        sorted_score = np.argsort(u_score_list)
        index_most_uncertain = unlabeled_indices[sorted_score[:5]]
        u_score_list = 1 - np.array(u_score_list)

    return index_most_uncertain


class ActiveLearner(object):
    """
    class for active learning.
    Args:
        estimator: a function object that implements fit(), and either
        predict() for supervised learning or label_distributions_() for semi-
        supervised learning.
        X_training: Either a list of rdkit.Chem.Mol's or a 2D array-like
        object.
        Y_training: a 2D array-like object that stores the labels for
        each data point in X_training. Each data point can contain results
        for multiple tasks.
        finger_print_fn: a function that maps a Mol object to a 1d numpy arrray.
        query_strategy: Sampling method. One of ['E', 'LC', 'MS', 'RS']
        for entropy, least confidence, margin sampling, and
        random sampling respectively.
    """

    def __init__(self,
                 estimator,
                 X_training,
                 Y_training,
                 finger_print_fn=None,
                 query_strategy='E'):
        self.X_training_mols = None
        if finger_print_fn is None:
            finger_print_fn = lambda x: AllChem.GetMorganFingerprintAsBitVect(
                x, 2, nBits=512)
        self.finger_print_fn = finger_print_fn
        if all([isinstance(x, Mol) for x in X_training]):
            self.X_training_mols = copy.deepcopy(X_training)
            X_training_vectors = [finger_print_fn(mol) for mol in X_training]
            self.X_training_vectors = np.array(X_training_vectors)
        elif len(np.array(X_training).shape) != 2:
            raise ValueError("X_training should be either a 2d array, "\
                             "or a list of rdkit.Chem.Mol's.")
        else:
            self.X_training_vectors = np.array(X_training)

        self.Y_training = np.array(Y_training)
        self.estimator = estimator
        self.query_strategy = query_strategy

    def query(self, X_pool):
        """
        Args:
            X_pool: Either a list of rdkit.Chem.Mol objects or a
            2D array-like
            object. A datapoint is picked from this pool of data points.
        """

        if self.query_strategy == 'RS':
            query_idx = np.random.randint(len(X_pool))
            return query_idx, X_pool[query_idx]

        estimators = [
            copy.deepcopy(self.estimator)
            for task in range(self.Y_training.shape[1])
        ]

        if all([isinstance(x, Mol) for x in X_pool]):
            X_pool_mols = X_pool
            X_pool_vectors = np.array([self.finger_print_fn(mol) for mol in X_pool])
        elif len(np.array(X_pool).shape) != 2:
            raise ValueError("X_pool should be either a 2d array-like"\
                             "object, or a list of rdkit.Chem.Mol's.")
        else:
            X_pool_vectors = np.array(X_pool)
        predictions = []
        if isinstance(self.estimator, (LabelPropagation, LabelSpreading)):
            X = np.vstack([self.X_training_vectors, X_pool_vectors])
            append_shape = (len(X_pool_vectors), self.Y_training.shape[1])
            Y = np.vstack([self.Y_training, np.ones(append_shape) * -1])
            ss = StandardScaler()
            X_norm = ss.fit_transform(X)
            for i, estimator in enumerate(estimators):
                estimator.fit(X_norm, Y[:, i])
                predictions.append(
                    estimator.label_distributions_[-len(X_pool_vectors):])
        else:
            for task, estimator in enumerate(estimators):
                ss = StandardScaler()
                present = (self.Y_training[:, task] != -1)
                X_training_norm = ss.fit_transform(self.X_training_vectors[present])
                estimator.fit(X_training_norm,
                              self.Y_training[:, task][present])
                predictions.append(estimator.predict_proba(ss.transform(X_pool_vectors)))
        if self.query_strategy == 'E':
            pred_entropies_list = []
            for task in range(len(estimators)):
                entropies = stats.distributions.entropy(predictions[task].T)
                pred_entropies_list.append(entropies)
            pred_entropies = np.vstack(pred_entropies_list, )
            pred_entropies = np.sum(pred_entropies, axis=0)
            uncertainty_score_list = pred_entropies / np.max(pred_entropies)
            sorted_entropies = np.argsort(pred_entropies)
            query_idx = sorted_entropies[-1]

        elif self.query_strategy == 'LC':
            confidences = [
                np.max(prediction, axis=1) for prediction in predictions
            ]
            confidence_gathered = np.mean(confidences, axis=0)
            query_idx = np.argmin(confidence_gathered)

        elif self.query_strategy == 'MS':
            margins = []
            for prediction in predictions:
                confidence_sorted = np.sort(prediction, axis=1)
                if prediction.shape[1] == 1:
                    margin = confidence_sorted[:, 0]
                else:
                    margin = confidence_sorted[:, -1] - confidence_sorted[:, -2]
                margins.append(margin)
            margins = np.sum(margins, axis=0)
            query_idx = np.argmin(margins)
        query_inst = X_pool[query_idx]
        return query_idx, query_inst

    def teach(self, x_new, y_new):
        """
        Updates the training data with a new data point.
        Args:
            x_new: Either an rdkit.Chem.Mol object or a 1D
            array-like object.
            y_new: A 1D array-like object.
        """
        X_training_mols = None
        if self.X_training_mols is not None:
            if not isinstance(x_new, Mol):
                raise ValueError("x_new should be a molecule")
            X_training_mols = self.X_training_mols + [x_new]
            X_training_vectors = np.vstack(
                [self.X_training_vectors,
                 self.finger_print_fn(x_new)])
        else:
            X_training_vectors = np.vstack([self.X_training_vectors, x_new])
        Y_training = np.vstack([self.Y_training, y_new])
        self.X_training_mols = X_training_mols
        self.X_training_vectors = X_training_vectors
        self.Y_training = Y_training

    def query_and_teach_n_times(self, n, X_pool, Y_pool):
        """
        Repeat querying and teaching with pooled data, and labels.
        Args:
            n (int): the number of iteration. -1 means all pooled
            data points are queried.
            X_pool: Either a list of rdkit.Chem.Mol objets, or 2D array
            -like object.
            Y_pool: A 2D array_like object (num_datapoints, num_tasks)
            that stores the labels for X_pool. -1 means the data for
            the corresponding datapoint and task is missing.
        Returns:
            query_idx: the indices used for querying and sampling.
        """
        query_idxs = []
        if n == -1:
            n = len(X_pool)
        X_pool = copy.deepcopy(X_pool)
        ind_list = list(range(len(X_pool)))
        logging.info(f"number of data points in X_pool: {len(X_pool)}")
        for nn in range(n):
            logging.info(f"iteration num: {nn} out of {n}.")
            idx, inst = self.query(X_pool)
            new_data_point = X_pool[idx]
            new_label = Y_pool[idx]
            self.teach(new_data_point, new_label)
            if isinstance(new_data_point, Mol):
                X_pool.pop(idx)
            else:
                X_pool = np.vstack([X_pool[:idx], X_pool[idx + 1:]])

            Y_pool = np.vstack([Y_pool[:idx], Y_pool[idx + 1:]])
            query_idxs.append(ind_list.pop(idx))
        return query_idxs
