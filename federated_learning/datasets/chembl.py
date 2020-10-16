#!/usr/bin/env
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


def load_data(max_n_atoms, max_n_types, protein_max_seqlen, n_groups):
    train_dataset = ChemblDataset(max_n_atoms, max_n_types, protein_max_seqlen, n_groups=n_groups)
    return train_dataset


class ChemblDataset(ClientData):
    _urls = {'url': 'https://github.com/clinfo/kGCN/files/5362776/CheMBL_MMP_table_data.zip',
             'filename': 'CheMBL_MMP_table_data.zip',
             'csvfilename': 'dataset_benchmark.tsv'}
    def __init__(self, 
                 max_n_atoms=150, max_n_types=100, protein_max_seqlen=750, n_groups=2,
                 savedir='./data', smiles_canonical=True):
        self.savedir = savedir
        self.filename = savedir + '/' + self._urls['filename'].replace('.zip', '')
        _download(self._urls['url'], self._urls['filename'], savedir)
        extract_zipfile(self.savedir + '/' + self._urls['filename'], savedir)
        self.max_n_atoms = max_n_atoms
        self.max_n_types = max_n_types
        self.protein_max_seqlen = protein_max_seqlen
        self.n_groups = n_groups
        self.smiles_canonical = smiles_canonical
        self._data = self._read_tsv(self.savedir + '/' + self._urls['csvfilename'], skiprows=0)
        self.salt_remover = SaltRemover.SaltRemover()
        self._client_ids = sorted(create_ids(n_groups, 'CHEMBL'))
        self._adj_shape = (max_n_atoms, max_n_atoms)
        self._feature_shape = (max_n_atoms, max_n_types)
        # # assign _client_id to data.
        self.data = {_id: [] for _id in self._client_ids}
        counter = 0
        for data, _id in zip(self._data.itertuples(), np.random.choice(self._client_ids, len(self._data))):
            self.data[_id].append(self._create_element(data))
            if counter > 10:
                break
            counter += 1
        g = tf.Graph()
        with g.as_default():
            tf_dataset = self._create_dataset(self._client_ids[0])
            self._element_type_structure = tf_dataset.element_spec

    def _create_element(self, data):
        label = np.array(data[1], dtype=np.int32)
        mol = Chem.MolFromSmiles(data[4])
        if self.smiles_canonical:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(smiles)
        features = pad_bottom_matrix(self._create_mol_feature(mol), self.max_n_atoms)
        adjs = pad_bottom_right_matrix(rdmolops.GetAdjacencyMatrix(mol), self.max_n_atoms)
        res = self.salt_remover.StripMol(mol, dontRemoveEverything=True)
        protein_seq = self._create_one_hot_protein_seq(data[5])
        return ({'adjs': adjs,
                 'features': features,
                 'protein_seq': protein_seq}, label)

    def _create_one_hot_protein_seq(self, protein_seq: str):
        one_letter_aa = 'XACDEFGHIKLMNPQRSTVWY'
        vec = []
        for idx in range(self.protein_max_seqlen):
            if len(protein_seq) > idx:
                c = protein_seq[idx]
                vec.append(one_letter_aa.index(c))
            else:
                vec.append(one_letter_aa.index('X'))
        return vec

    def _create_mol_feature(self, mol):
        mol_features = np.array([m.GetAtomicNum() for m in mol.GetAtoms()])
        mol_features = one_hot(mol_features, self.max_n_types).astype(np.int32)
        return mol_features

    def _create_dataset(self, client_id):
        # https://stackoverflow.com/questions/52582275/tf-data-with-multiple-inputs-outputs-in-keras
        _data = collections.OrderedDict({key: [] for key in sorted(['adjs', 'features', 'protein_seq'])})
        _labels = []
        for idx, mol in enumerate(self.data[client_id]):
            adjs = np.expand_dims(mol[0]['adjs'], axis=0) # for adj channel
            _data['adjs'].append(adjs)
            _data['features'].append(mol[0]['features'])
            _data['protein_seq'].append(mol[0]['protein_seq'])
            _labels.append(mol[1])
        _data = collections.OrderedDict((name, np.array(ds))
                                        for name, ds in sorted(_data.items()))
        return tf.data.Dataset.from_tensor_slices((_data, np.array(_labels)))

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

    def _read_tsv(self, filename, skiprows=None):
        df = pd.read_csv(filename, delimiter='\t')
        return df

    def __len__(self):
        return self._len


if __name__ == "__main__":
    c = ChemblDataset()
