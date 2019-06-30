import os
import tempfile
import subprocess
from pathlib import Path

import pytest

multi_modal_profeat_urls = [
    'https://gitlab.com/koji.ono/gcn_data/raw/master/multi_modal_profeats/mol_00001.pkl',
    'https://gitlab.com/koji.ono/gcn_data/raw/master/multi_modal_profeats/mol_00002.pkl',
    'https://gitlab.com/koji.ono/gcn_data/raw/master/multi_modal_profeats/mol_00003.pkl'    
    ]

def download(url, filename):
    urllib.request.urlretrieve(url, filename)    

@pytest.fixture()
def cleandir():
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)

@pytest.fixture()
def multi_modal_profeat(cleandir):
    for u in multi_modal_profeat_urls:
        filename = u.split('/')[-1]
        subprocess.run([f'curl -s --header "Private-Token: crfEYvKXCnTXzNaq5xXR" {u} -O {filename}'], shell=True)
    return os.listdir(os.getcwd())
