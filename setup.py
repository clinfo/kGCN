import setuptools
import shutil
import os
from pathlib import Path
path=os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(path+"/gcn.py", path+"/kgcn/gcn.py")
shutil.copyfile(path+"/gcn_gen.py", path+"/kgcn/gen.py")
shutil.copyfile(path+"/script_cv/cv_splitter.py", path+"/kgcn/cv_splitter.py")
shutil.copyfile(path+"/opt_hyperparam.py", path+"/kgcn/opt.py")

setuptools.setup(
    name="kGCN",
    version="1.0",
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="graph convolutional network library",
    long_description="graph convolutional network library",
    long_description_content_type="text/markdown",
    url="https://github.com/clinfo/kGCN",
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts' : [
            'kgcn = kgcn.gcn:main',
            'kgcn-chem = kgcn.preprocessing.chem:main',
            'kgcn-cv-splitter = kgcn.cv_splitter:main',
            'kgcn-opt = kgcn.opt:main',
            'kgcn-gen = kgcn.gen:main',],
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
