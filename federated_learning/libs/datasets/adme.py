#!/usr/bin/env python
# coding: utf-8
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
from ..utils import extract_zipfile, create_ids, one_hot, pad_bottom_right_matrix, pad_bottom_matrix


def load_data(task, max_n_atoms, max_n_types, n_groups, subset_ratios: list or int=None, loaddir: str='./data'):
    """loads the federated tox21 dataset.
    """
    dataset = ADMEDataset(max_n_atoms, max_n_types, n_groups, subset_ratios)
    return dataset

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
    def __init__(self, max_n_atoms=150, max_n_types=100, n_groups=2,
                 subset_ratios=None, none_label=None, loaddir='./data'):
        pass
        
