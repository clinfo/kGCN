# GCNVisualizer

This library is for visualizing for most effective input features.

## Requirements
* python >= 3.6
* matplotlib >= 2.0.0
* networkx

### optional
* bioplot >= 0.0.2
* ipython

## Installation

```shell
$ pip install -r requirements.txt
$ pip install -e .
```

## How to prepare input files

```shell
$ kgcn visualize --config <config>
```

## How to use

```shell
$ gcnv -i hoge.pkl
```

# Reference

### Axiomatic Attribution for Deep Networks(2017)

* http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf

### An Investigation of Uncertainty and Sensitivity Analysis Techniques for Computer Models

* http://mycourses.ntua.gr/courses/CIVIL1086/document/WRM_Part_B_Makropoulos/iman1988.pdf~

### "Why Should I Trust You?": Explaining the Predictions of Any Classifier

* https://arxiv.org/pdf/1602.04938
* https://github.com/marcotcr/lime
