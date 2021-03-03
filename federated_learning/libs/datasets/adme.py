#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import collections

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import SaltRemover
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import ClientData

from ..utils import download as _download
from ..utils import (extract_zipfile, create_ids, one_hot, pad_bottom_right_matrix,
                     pad_bottom_matrix, create_mol_feature)


def load_data(FL_FLAG, datapath, task_name, max_n_atoms, max_n_types, n_groups,
              subset_ratios: list or int=None):
    """loads the federated tox21 dataset.
    """
    
    if FL_FLAG:
        dataset = ADMEDataset(datapath, task_name, max_n_atoms, max_n_types, n_groups, subset_ratios)
    else:
        dataset = create_dataset(datapath, task_name, max_n_atoms, max_n_types)
    return dataset

def _create_element(mol, label, max_n_atoms, max_n_types, salt_remover):
    label = np.array(label)
    salt_removed_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    features = pad_bottom_matrix(
        create_mol_feature(salt_removed_mol, max_n_types), max_n_atoms)
    adjs = pad_bottom_right_matrix(
        rdmolops.GetAdjacencyMatrix(salt_removed_mol), max_n_atoms)
    return ({'adjs': adjs, 'features': features}, label)

def _read_sdf_file(datapath, task_name):
    sdf_name = _task_name_to_sdf_name(task_name)
    file_path = Path(datapath) / sdf_name
    mols = Chem.SDMolSupplier(str(file_path), strictParsing=False)
    data = []
    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            rdmolops.GetAdjacencyMatrix(mol)
        except Exception as e:
            continue
        data.append(mol)
    return data

def _read_dataset_file(datapath, task_name):
    salt_remover = SaltRemover.SaltRemover()
    activity_data = _read_activity_xml_file(datapath, task_name)
    task_type = task_name.split('_')[-1]
    task_target = task_name.split('_')[-2]
    label_column = 'value' if task_type == 'reg' else 'value.class'
    if task_target == 'fe':
        label_column = 'value'
    
    mols = []
    labels = []
    fe_threshfold = 0.3
    for data in activity_data[['inchi', label_column]].to_numpy():
        label = data[1]
        if task_target == 'fe':
            if label > fe_threshfold:
                label = 'High'
            else:
                label = 'Low'
        if label == 'High':
            label = 1
        elif label == 'Low':
            label = 0
        elif label == 'Medium':
            label = 2
        elif isinstance(label, float):
            pass
        elif np.isnan(label):
            continue
        else:
            raise Exception("label not found error")

        m = Chem.inchi.MolFromInchi(data[0])
        m = salt_remover.StripMol(m, dontRemoveEverything=True)
        mols.append(m)
        if task_type == 'cls':
            num_classes = 2            
            if task_target == 'llc':
                num_classes = 3
            label = one_hot(label, num_classes)
        else:
            # regression
            label = tf.math.log(label+1e-12)
        labels.append(label)
    return mols, labels

def create_dataset(datapath, task_name, max_n_atoms, max_n_types):
    salt_remover = SaltRemover.SaltRemover()
    mols, labels = _read_dataset_file(datapath, task_name)
    elements = [_create_element(mol, label, max_n_atoms, max_n_types, salt_remover)
                for mol, label in zip(mols, labels)]
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

def _task_name_to_sdf_name(task_name):
    task_to_sdf = {'abso_sol_cls': 'Sol_514_standardized.sdf',
                   'abso_cacco_cls': 'Papp_caco2_4294_standardized.sdf',
                   'abso_cacco_reg': 'Papp_caco2_4294_standardized.sdf',
                   'abso_llc_reg': 'Papp_LLC-human_461_standardized.sdf',
                   'dist_llc_cls': 'NER_LLC-human_445_standardized.sdf',
                   'dist_fu_man_reg': 'fubrain_580_standardized.sdf',
                   'dist_fu_hum_reg': 'fup_human_2559_standardized.sdf',
                   'dist_fu_rat_reg': 'fup_rat_539_standardized.sdf',
                   'dist_rb_rat_reg': 'Rb_rat_162_standardized.sdf',
                   'meta_clint_reg': 'CLint_human_5216_standardized.sdf',
                   'excr_fe_cls': 'fe_human_340_standardized.sdf'}
    return task_to_sdf[task_name]

def _task_name_to_sheet_name(task_name):
    task_to_sheet = {'abso_sol_cls': 'Sol7.4',
                     'abso_cacco_cls': 'Papp(AtoB)_Caco-2',
                     'abso_cacco_reg': 'Papp(AtoB)_Caco-2',
                     'abso_llc_reg': 'Papp(AtoB)_LLC-human',
                     'dist_llc_cls': 'NER_LLC-human',
                     'dist_fu_man_reg': 'fubrain',
                     'dist_fu_hum_reg': 'fup_human',
                     'dist_fu_rat_reg': 'fup_rat',
                     'dist_rb_rat_reg': 'Rb_rat',
                     'meta_clint_reg': 'CLint_human',
                     'excr_fe_cls': 'fe_human'}
    return task_to_sheet[task_name]

def _read_activity_xml_file(datapath, task_name):
    # version 1: 20201016_activity_dataset.xlsx
    # version 2: 20201213_activity_dataset.xlsx
    file_path = Path(datapath) / '20201213_activity_dataset.xlsx'
    sheet_name = _task_name_to_sheet_name(task_name)
    return pd.read_excel(file_path, sheet_name=sheet_name)

class ADMEDataset(ClientData):
    # ADME_Activity and SDFs
    _files = {
        'activity': '20201016_activity_dataset.xlsx',
        'clint_human': 'CLint_human_5216_standardized.sdf',
        'ner_llc-human': 'NER_LLC-human_445_standardized.sdf',
        'papp_llc-human': 'Papp_LLC-human_461_standardized.sdf',
        'papp_caco2': 'Papp_caco2_4294_standardized.sdf',
        'rb_rat': 'Rb_rat_162_standardized.sdf',
        'sol': 'Sol_514_standardized.sdf',
        'fe_human': 'fe_human_340_standardized.sdf',
        'fubrain': 'fubrain_580_standardized.sdf',
        'fup_human': 'fup_human_2559_standardized.sdf',
        'fup_rat': 'fup_rat_539_standardized.sdf'
        }
    def __init__(self, datapath, task_name, max_n_atoms=150, max_n_types=100, n_groups=2,
                 subset_ratios=None, none_label=None, loaddir='./data'):
        mols, labels = _read_dataset_file(datapath, task_name)
        self.task_name = task_name
        self.max_n_atoms = max_n_atoms
        self.max_n_types = max_n_types
        self.n_groups = n_groups
        self._len = len(labels)
        self._salt_remover = SaltRemover.SaltRemover()
        self._client_ids = sorted(create_ids(n_groups, 'ADME'))
        self._adj_shape = (max_n_atoms, max_n_atoms)
        self._feature_shape = (max_n_atoms, max_n_types)

        self.elements = collections.defaultdict(list)
        if subset_ratios is None:
            self.subset_ratios = [
                1. / self.n_groups for _ in range(self.n_groups)]
        else:
            self.subset_ratios = subset_ratios
        print(self._client_ids, self._len, self.subset_ratios)
        for (mol, label), client_id in zip(zip(mols, labels),
                                           np.random.choice(self._client_ids, self._len, p=self.subset_ratios)):
            self.elements[client_id].append(
                _create_element(mol, label, self.max_n_atoms, self.max_n_types, self._salt_remover))

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
    
