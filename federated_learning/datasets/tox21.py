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
from tensorflow_federated.python.simulation import hdf5_client_data, ClientData


from .utils import download as _download
from .utils import extract_zipfile, create_ids, one_hot, pad_bottom_right_matrix, pad_bottom_matrix

        
def load_data(savedir='./data', n_groups=2):
    """loads the federated tox21 dataset.
    """
    train_dataset = Tox21Dataset('train', n_groups=n_groups)
    # Tox21Dataset('val').to_csv('tox21_val.csv')
    val_dataset = Tox21Dataset('val')
    #train_dataset.to_csv('tox21_train.csv')
    #test_dataset = Tox21Dataset('test')
    return train_dataset, val_dataset

class Tox21Dataset(ClientData):
    edge_types = {rdkit.Chem.rdchem.BondType.SINGLE: 0,
                  rdkit.Chem.rdchem.BondType.DOUBLE: 1,
                  rdkit.Chem.rdchem.BondType.TRIPLE: 2,
                  rdkit.Chem.rdchem.BondType.AROMATIC: 3}
    _urls = {
        'train': {
            'url': 'https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf',
            'filename': 'tox21_10k_data_all.sdf.zip'},
        'val': {
            'url': 'https://tripod.nih.gov/tox21/challenge/download?'
            'id=tox21_10k_challenge_testsdf',
            'filename': 'tox21_10k_challenge_test.sdf.zip'
        },
        'test': {
            'url': 'https://tripod.nih.gov/tox21/challenge/download?'
            'id=tox21_10k_challenge_scoresdf',
            'filename': 'tox21_10k_challenge_score.sdf.zip'
        }
    }
    _columns = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                    'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                    'SR-HSE', 'SR-MMP', 'SR-p53', 'mol_id', 'smiles']
    _label_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                    'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                    'SR-HSE', 'SR-MMP', 'SR-p53']
    
    def __init__(self, target='train', savedir='./data',
                 none_label=None, max_n_atoms=150, max_n_types=100, n_groups=2):
        self.target = target
        self.savedir = savedir
        self.filename = savedir + '/' + self._urls[self.target]['filename'].replace('.zip', '')
        _download(self._urls[self.target]['url'], self._urls[self.target]['filename'], savedir)
        extract_zipfile(self.savedir + '/' + self._urls[self.target]['filename'], savedir)
        self.none_label = none_label
        self.max_n_atoms = max_n_atoms
        self.max_n_types = max_n_types
        self.n_groups = n_groups
        self.mols = self._get_valid_mols()
        self.salt_remover = SaltRemover.SaltRemover()
        self._client_ids = sorted(create_ids(n_groups, 'TOXG'))
        self._adj_shape = (max_n_atoms, max_n_atoms)
        self._feature_shape = (max_n_atoms, max_n_types)
        # assign _client_id to data.
        self.data = {_id: [] for _id in self._client_ids}
        for data, _id in zip(self.mols, np.random.choice(self._client_ids, len(self.mols))):
            self.data[_id].append(self._create_element(self.mols[data]))

        # Get the types and shapes from the first client. We do it once during
        # initialization so we can get both properties in one go.

        g = tf.Graph()
        with g.as_default():
            tf_dataset = self._create_dataset(self._client_ids[0])
            self._element_type_structure = tf_dataset.element_spec

    def _create_element(self, mol):
        label = np.array(self._get_label(mol), dtype=np.float)
        mask_label = np.invert(np.isnan(label))
        features = pad_bottom_matrix(self._create_mol_feature(mol), self.max_n_atoms)
        dense_adj = pad_bottom_right_matrix(rdmolops.GetAdjacencyMatrix(mol), self.max_n_atoms)
        res = self.salt_remover.StripMol(mol, dontRemoveEverything=True)
        return {'label': label,
                'mask_label': mask_label,
                'features': features,
                'dense_adj': dense_adj}

    def _create_mol_feature(self, mol):
        mol_features = np.array([m.GetAtomicNum() for m in mol.GetAtoms()])
        mol_features = one_hot(mol_features, self.max_n_types).astype(np.int32)
        return mol_features

    def _create_dataset(self, client_id):
        _data = collections.OrderedDict({key: [] for key in sorted(self.data[client_id][0].keys())})
        for idx, mol in enumerate(self.data[client_id]):
            _data['label'].append(mol['label'])
            _data['mask_label'].append(mol['mask_label'])
            _data['features'].append(mol['features'])
            _data['dense_adj'].append(mol['dense_adj'])
            # FIXME: store sparse matrix to reduce memory usage.
        _data = collections.OrderedDict((name, np.array(ds))
                                        for name, ds in sorted(_data.items()))
        return tf.data.Dataset.from_tensors(_data)

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

    def create_tf_dataset_for_client(self, client_id):
        if client_id not in self.client_ids:
            raise ValueError(
                "ID [{i}] is not a client in this ClientData. See "
                "property `client_ids` for the list of valid ids.".format(
                    i=client_id))
        tf_dataset = self._create_dataset(client_id)
        # tensor_utils.check_nested_equal(tf_dataset.element_spec,
        #                                 self._element_type_structure)
        return tf_dataset
    
    def _get_valid_mols(self):
        tmpmols = Chem.SDMolSupplier(self.filename, strictParsing=False)
        mols = {}
        for mol in tmpmols:
            if mol is None:
                continue
            try:
                rdmolops.GetAdjacencyMatrix(mol)
            except Exception as e:
                continue
            edge_index, _ = self.get_mol_edge_index(mol, self.edge_types)
            if edge_index.size == 0:
                continue
            if mol.HasProp('DSSTox_CID'):
                key = 'TOX' + mol.GetProp('DSSTox_CID')
            else:
                key = mol.GetProp('_Name')                
            if not key in mols.keys():
                mols[key] = mol
            else:
                for prop in mol.GetPropNames():
                    if prop in self._label_names:
                        mols[key].SetProp(prop, mol.GetProp(prop))
        return mols
    
    def _get_label(self, mol: Chem):
        labels = []
        for label in self._label_names:
            if mol.HasProp(label):
                labels.append(int(mol.GetProp(label)))
            else:
                labels.append(self.none_label)
        return labels

    def get_mol_edge_index(self, mol, edge_types: dict):
        row, col, bond_idx = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_idx += 2 * [edge_types[bond.GetBondType()]]
        edge_index = np.array([row, col], dtype=np.int32)
        num_classes = len(edge_types)        
        edge_attr = one_hot(bond_idx, num_classes)
        return edge_index, edge_attr

    def read_csv(self, filename):
        pass

    def to_csv(self, filename='tox21.csv'):
        data = []
        for idx, m in enumerate(self.mols):
            mol = self.mols[m]
            res = self.salt_remover.StripMol(mol, dontRemoveEverything=True)
            data.append([*self._get_label(res),
                         'TOX' + res.GetProp('DSSTox_CID'),
                         Chem.MolToSmiles(res)])
        df = pd.DataFrame(data, columns=self._columns, dtype=int)
        df.to_csv(filename, index=False)

    def __len__(self):
        return self._len
