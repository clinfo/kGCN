#!/usr/bin/env python
# coding: utf-8
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

from typing import Optional, Set, List, Dict


def load_data(FL_FLAG, dataset_name, n_groups=2, subset_ratios: list = None):
    if FL_FLAG:
        dataset = ToxicityDataset(dataset_name, n_groups, subset_ratios)
    else:
        dataset = create_dateset(dataset_name)
    return dataset


def _read_dataset_file(dataset_name: str):
    file_path = os.path.join('data', 'tox_AMES_summary.20201106.xlsx')
    dataset = pd.read_excel(file_path, sheet_name=dataset_name)
    return filter_dataset(dataset, 60)  # TODO 引数で指定する。


def filter_dataset(dataset, max_n_atoms):
    validities = [Chem.MolFromSmiles(
        data.Canonical_SMILES).GetNumAtoms() <= max_n_atoms for data in dataset.itertuples()]
    return dataset[validities]


def _create_mol_feature(mol, atom_num_to_index: Dict[int, int]) -> np.array:
    """
    creates a feature vector from atoms in a molecule

    mol: `Chem.rdchem.Mol`
    all_atom_nums: set of atoms in the dataset
    """

    atom_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    max_n_types = len(atom_num_to_index)
    mol_features = one_hot([atom_num_to_index[atom_num]
                            for atom_num in atom_nums], max_n_types).astype(np.int32)
    return mol_features


def _get_all_atom_nums(data: pd.DataFrame, salt_remover: SaltRemover) -> Set[int]:
    """
    data: dataframe represents the dataset
    """
    mols = [Chem.MolFromSmiles(data_row.Canonical_SMILES)
            for data_row in data.itertuples()]
    salt_removed_mols = [salt_remover.StripMol(
        mol, dontRemoveEverything=True) for mol in mols]
    atom_nums = set()
    for mol in salt_removed_mols:
        atom_nums |= {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    return atom_nums


def _create_element(data_row, max_n_atoms, salt_remover, atom_num_to_index):
    """
    creates an element for `tf.Dataset`

    data_row: an element of an iterator obtained by `data.itertuples`
    """
    label = np.array(1 if data_row.Label == "Positive" else 0, dtype=np.int32)
    mol = Chem.MolFromSmiles(data_row.Canonical_SMILES)
    salt_removed_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    features = pad_bottom_matrix(
        _create_mol_feature(salt_removed_mol, atom_num_to_index), max_n_atoms)
    adjs = pad_bottom_right_matrix(
        rdmolops.GetAdjacencyMatrix(salt_removed_mol), max_n_atoms)
    return ({'adjs': adjs, 'features': features}, label)


def create_dateset(dataset_name):
    salt_remover = SaltRemover.SaltRemover()
    data = _read_dataset_file(dataset_name)
    max_n_atoms = max(Chem.MolFromSmiles(data_row.Canonical_SMILES).GetNumAtoms()
                      for data_row in data.itertuples())
    all_atom_nums = _get_all_atom_nums(data, salt_remover)
    atom_num_to_index = {atom_num: idx for idx,
                         atom_num in enumerate(all_atom_nums)}
    elements = [_create_element(data_row, max_n_atoms, salt_remover, atom_num_to_index)
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

    def __init__(self, dataset_name, n_groups=2, subset_ratios=None):
        self._data = _read_dataset_file(dataset_name)
        self._salt_remover = SaltRemover.SaltRemover()
        self._all_atom_nums = _get_all_atom_nums(
            self._data, self._salt_remover)
        self._atom_num_to_index = {atom_num: idx for idx,
                                   atom_num in enumerate(self._all_atom_nums)}
        self.max_n_atoms = max(Chem.MolFromSmiles(data_row.Canonical_SMILES).GetNumAtoms()
                               for data_row in self._data.itertuples())
        self.max_n_types = len(self._atom_num_to_index)
        self.n_groups = n_groups
        self._len = len(self._data)
        self._client_ids = sorted(create_ids(n_groups, 'TOX'))
        self._adj_shape = (self.max_n_atoms, self.max_n_atoms)
        self._feature_shape = (self.max_n_atoms, self.max_n_types)

        if subset_ratios is None:
            self.subset_ratios = [
                1. / self.n_groups for _ in range(self.n_groups)]
        else:
            self.subset_ratios = subset_ratios

        self.elements = collections.defaultdict(list)
        for data_row, client_id in zip(self._data.itertuples(), np.random.choice(self._client_ids, self._len, p=self.subset_ratios)):
            self.elements[client_id].append(
                _create_element(data_row, self.max_n_atoms, self._salt_remover, self._atom_num_to_index))

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

    @ property
    def adj_shape(self):
        return self._adj_shape

    @ property
    def feature_shape(self):
        return self._feature_shape

    @ property
    def client_ids(self):
        return self._client_ids

    @ property
    def element_type_structure(self):
        return self._element_type_structure

    @ property
    def dataset_computation(self):
        raise NotImplementedError("tox21:dataset_computation")
