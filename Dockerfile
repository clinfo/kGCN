FROM nvidia/cuda:9.0-cudnn7-runtime
MAINTAINER Kojima <kojima.ryosuke.8e@kyoto-u.ac.jp>
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y wget bzip2 curl git
RUN mkdir -p ~/usr/gcn

WORKDIR /usr/gcn
ENV CONDA_ROOT /root/miniconda
ENV PATH /root/miniconda/bin:$PATH
SHELL ["/bin/bash", "-c"]

ARG GIT_USER
ARG GIT_TOKEN
RUN echo ${GIT_USER}:${GIT_TOKEN}
RUN git clone https://${GIT_USER}:${GIT_TOKEN}@github.com/clinfo/GraphCNN.git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_ROOT && \
    ln -s ${CONDA_ROOT}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDA_ROOT}/etc/profile.d/conda.sh" >> ~/.bashrc 
RUN pip install tensorflow-gpu
RUN conda install scikit-learn joblib pandas keras
RUN conda install rdkit -c rdkit
RUN apt install -y libfontconfig1 libxrender1
