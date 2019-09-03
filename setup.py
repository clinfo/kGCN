import setuptools
import shutil
import os
from pathlib import Path
path=os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(path+"/layers.py", path+"/gcn_modules/layers.py")
if not os.path.isdir(path+"/kgcn"):
    shutil.copytree(path+"/gcn_modules",path+"/kgcn")
Path(path+"/kgcn/__init__.py").touch()

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
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
