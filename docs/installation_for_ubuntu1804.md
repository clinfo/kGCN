### Installation (for Ubuntu 18.04)

First, please install anaconda by the anaconda instruction.
```
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
sh ./Anaconda3-2019.07-Linux-x86_64.sh
```

Next, please install following libraries.
```
source ~/.bashrc
conda update conda
conda install joblib

# for CPU-only
conda install tensorflow==1.15.0
# with GPU support
conda install tensorflow-gpu==1.15.0
```

Finally, please install kGCN
```
pip install --upgrade git+https://github.com/clinfo/kGCN.git
```

Sample programs can be downloaded by git clone from this repositpry:
```
git clone https://github.com/clinfo/kGCN.git
```

Optional library
```
sudo apt install -y libfontconfig1 libxrender1
conda install rdkit -c rdkit
```

