# coding: utf-8
import shutil
import logging
from pathlib import Path
from zipfile import ZipFile
from typing import List

import numpy as np
import requests



def create_client_data(source, n: int, batch_size: int, epochs: int):
    return source.create_tf_dataset_for_client(source.client_ids[n]).repeat(epochs).batch(batch_size)

def get_logger(name, level='DEBUG'):
    FORMAT = '%(asctime)-15s - %(pathname)s - %(funcName)s - L%(lineno)3d ::: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def download(url: str, filename: str, savedir: str):
    savedir = Path(savedir).expanduser().resolve()
    savefile = savedir / filename
    if not savedir.exists():
        savedir.mkdir()
    if not savefile.exists():
       with requests.get(url, stream=True) as r:
           with open(savefile, 'wb') as f:
               shutil.copyfileobj(r.raw, f)
    return savefile

def extract_zipfile(zfilename: str, extractdir: str ='.') -> List[str]:
    with ZipFile(zfilename) as zipfile:
        zipfile.extractall(extractdir)
        namelist = zipfile.namelist()
    return namelist

def create_ids(num, prefix='id'):
    return [f'{prefix}{i:03}' for i in range(num)]
    
def one_hot(values: np.array, n_classes: int):
    return np.eye(n_classes)[values].astype(np.int32)

def pad_bottom_right_matrix(matrix: np.array, max_dim):
    n_bottom_padding = max_dim - matrix.shape[0]    
    n_right_padding = max_dim - matrix.shape[1]
    return np.pad(matrix, [(0, n_bottom_padding), (0, n_right_padding)], 'constant')

def pad_bottom_matrix(matrix: np.array, max_dim):
    n_bottom_padding = max_dim - matrix.shape[0]
    return np.pad(matrix, [(0, n_bottom_padding), (0, 0)], 'constant')


