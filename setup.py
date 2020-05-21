import setuptools
import shutil
import os

path = os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(f"{path}/gcn.py", f"{path}/kgcn/gcn.py")
shutil.copyfile(f"{path}/gcn_gen.py", f"{path}/kgcn/gen.py")
shutil.copyfile(f"{path}/script_cv/cv_splitter.py", f"{path}/kgcn/cv_splitter.py")
shutil.copyfile(f"{path}/opt_hyperparam.py", f"{path}/kgcn/opt.py")
shutil.copyfile(f"{path}/task_sparse_gcn.py", f"{path}/kgcn/task_sparse_gcn.py")

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
    entry_points={
        'console_scripts': [
            'kgcn = kgcn.gcn:main',
            'kgcn-chem = kgcn.preprocessing.chem:main',
            'kgcn-kg = kgcn.preprocessing.kg:main',
            'kgcn-cv-splitter = kgcn.cv_splitter:main',
            'kgcn-opt = kgcn.opt:main',
            'kgcn-gen = kgcn.gen:main',
            'kgcn-sparse = kgcn.task_sparse_gcn:main'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
