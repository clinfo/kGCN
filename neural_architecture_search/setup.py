import os
import re
import setuptools
from pathlib import Path


p = Path(__file__)

setuptools.setup(
    name="dbonas",
    version="0.1.0",
    python_requires='>3.7',
    author="Koji Ono",
    author_email="kbu94982@gmail.com",
    description="Deep Bayes Optimization Neural Architecture Search (DBONAS)",
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=setuptools.find_packages(),
    install_requires=['requests',],
    setup_requires=['numpy', 'pytest-runner'],
    tests_require=['pytest-cov', 'pytest-html', 'pytest'],
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
)
