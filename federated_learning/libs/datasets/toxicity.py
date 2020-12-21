#!/usr/bin/env python
# coding: utf-8
from typing import Dict, List
import collections
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import ClientData

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import SaltRemover
from ..utils import create_ids, one_hot, pad_bottom_right_matrix, pad_bottom_matrix


def load_data(FL_FLAG, dataset_name, max_n_atoms, max_n_types, n_groups=2, subset_ratios: list = None):
    if FL_FLAG:
        dataset = ToxicityDataset(
            dataset_name, max_n_atoms, max_n_types, n_groups, subset_ratios)
    else:
        dataset = create_dateset(dataset_name, max_n_atoms, max_n_types)
    return dataset


def _read_dataset_file(dataset_name: str):
    file_path = os.path.join('data', 'tox_AMES_summary.20201106.xlsx')
    return pd.read_excel(file_path, sheet_name=dataset_name)


def _create_mol_feature(mol, max_n_types):
    """
    creates a feature vector from atoms in a molecule

    mol: `Chem.rdchem.Mol`
    """
    mol_features = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    mol_features = one_hot(mol_features, max_n_types).astype(np.int32)
    return mol_features


def _create_element(data_row, max_n_atoms, max_n_types, salt_remover):
    """
    creates an element for `tf.Dataset`

    data_row: an element of an iterator obtained by `data.itertuples` 
    """
    label = np.array(1 if data_row.Label == "Positive" else 0, dtype=np.int32)
    mol = Chem.MolFromSmiles(data_row.Canonical_SMILES)
    salt_removed_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    features = pad_bottom_matrix(
        _create_mol_feature(salt_removed_mol, max_n_types), max_n_atoms)
    adjs = pad_bottom_right_matrix(
        rdmolops.GetAdjacencyMatrix(salt_removed_mol), max_n_atoms)
    return ({'adjs': adjs, 'features': features}, label)


def create_dateset(dataset_name, max_n_atoms, max_n_types):
    salt_remover = SaltRemover.SaltRemover()
    data = _read_dataset_file(dataset_name)
    elements = [_create_element(data_row, max_n_atoms, max_n_types, salt_remover)
                for data_row in data.itertuples()]
    inputs = collections.OrderedDict(
        {key: [] for key in sorted(['adjs', 'features'])})
    labels = []
    for mol, label in elements:
        adjs = np.expand_dims(mol['adjs'], axis=0)  # for adj channel
        inputs['adjs'].append(adjs)
        inputs['features'].append(mol['features'])
        labels.append(label)

    inputs = collections.OrderedDict(
        (name, np.array(ds)) for name, ds in sorted(inputs.items()))
    return tf.data.Dataset.from_tensor_slices((inputs, labels))


class ToxicityDataset(tff.simulation.ClientData):

    def __init__(self, dataset_name, max_n_atoms=250, max_n_types=100, n_groups=2, subset_ratios=None):
        self._data = _read_dataset_file(dataset_name)
        self.max_n_atoms = max_n_atoms
        self.max_n_types = max_n_types
        self.n_groups = n_groups
        self._len = len(self._data)
        self._salt_remover = SaltRemover.SaltRemover()
        self._client_ids = sorted(create_ids(n_groups, 'TOX'))
        self._adj_shape = (max_n_atoms, max_n_atoms)
        self._feature_shape = (max_n_atoms, max_n_types)

        if subset_ratios is None:
            self.subset_ratios = [
                1. / self.n_groups for _ in range(self.n_groups)]
        else:
            self.subset_ratios = subset_ratios

        self.elements = collections.defaultdict(list)
        for data_row, client_id in zip(self._data.itertuples(), np.random.choice(self._client_ids, self._len, p=self.subset_ratios)):
            self.elements[client_id].append(
                _create_element(data_row, self.max_n_atoms, self.max_n_types, self._salt_remover))

        g = tf.Graph()
        with g.as_default():
            tf_dataset = self._create_dataset(self._client_ids[0])
            self._element_type_structure = tf_dataset.element_spec

    def _create_dataset(self, client_id):
        # https://stackoverflow.com/questions/52582275/tf-data-with-multiple-inputs-outputs-in-keras
        _data = collections.OrderedDict(
            {key: [] for key in sorted(['adjs', 'features'])})
        _labels = []
        for mol, label in self.elements[client_id]:
            adjs = np.expand_dims(mol['adjs'], axis=0)  # for adj channel
            _data['adjs'].append(adjs)
            _data['features'].append(mol['features'])
            _labels.append(label)
        _data = collections.OrderedDict(
            (name, np.array(ds)) for name, ds in sorted(_data.items()))
        return tf.data.Dataset.from_tensor_slices((_data, np.array(_labels)))

    def create_tf_dataset_for_client(self, client_id):
        if client_id not in self.client_ids:
            raise ValueError(
                "ID [{i}] is not a client in this ClientData. See "
                "property `client_ids` for the list of valid ids.".format(i=client_id))
        tf_dataset = self._create_dataset(client_id)
        return tf_dataset

    @property
    def adj_shape(self):
        return self._adj_shape

    @property
    def feature_shape(self):
        return self._feature_shape

    @property
    def client_ids(self):
        return self._client_ids

    @property
    def element_type_structure(self):
        return self._element_type_structure

    @property
    def dataset_computation(self):
        raise NotImplementedError("tox21:dataset_computation")
