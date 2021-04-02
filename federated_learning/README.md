# Federated learning sample programs

# build envrionment

```shell
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
$ bash ~/miniconda.sh -b -p $HOME/miniconda
$ source $HOME/miniconda/bin/activate
$ conda env create –f environment.yml
$ conda activate fl
$ pip install -e ..
```

## install kGCN from `federated_learning` branch of `github.com/clinfo/kGCN` repo

```
$ pip install git+https://github.com/clinfo/kGCN.git@federated_learning
```


# Sample programs

## Tox21

```shell
$ python run_tox21.py 
```

### options

```shell
Usage: run_tox21.py [OPTIONS]

Options:
  --rounds INTEGER                the number of updates of the centeral model
  --clients INTEGER               the number of clients
  --epochs INTEGER                the number of training epochs in client
                                  traning.

  --batchsize INTEGER             the number of batch size.
  --lr FLOAT                      learning rate for the central model.
  --clientlr FLOAT                learning rate for client models.
  --model [gcn|gin]               support gcn or gin.
  --ratio TEXT                    set ratio of the biggest dataset in total
                                  datasize. Other datasets are equally
                                  divided. (0, 1)
  --task [NR-AR|NR-AR-LBD|NR-AhR|NR-Aromatase|NR-ER|NR-ER-LBD|NR-PPAR-gamma|SR-ARE|SR-ATAD5|SR-HSE|SR-MMP|SR-p53]
                                  set single task target. if not set,
                                  multitask is running.

  --help                          Show this message and exit.
```

## ChEBML

```shell
$ python run_chembl.py 
```

```shell
Usage: run_chembl.py [OPTIONS]

Options:
  --rounds INTEGER     the number of updates of the centeral model
  --clients INTEGER    the number of clients
  --epochs INTEGER     the number of training epochs in client traning.
  --batchsize INTEGER  the number of batch size.
  --lr FLOAT           learning rate for the central model.
  --clientlr FLOAT     learning rate for client models.
  --model TEXT         support gcn or gin.
  --ratio TEXT         set ratio of the biggest dataset in total datasize.
                       Other datasets are equally divided. (0, 1)
```

## toxicity

```shell
Usage: run_toxicity.py [OPTIONS]

Options:
  --mode [tuning|federated|cross_val|cross_val_flipped]
                                  "cross_val_flipped" simulates the situation
                                  where the single client only uses its own
                                  data. Other modes work namely

  --rounds INTEGER                the number of updates of the central model
  --clients INTEGER               the number of clients
  --epochs INTEGER                the number of training epochs in client
                                  traning.

  --dataset_name [Benchmark|NTP_PubChem_Bench.20201106|Ames_S9_minus.20201106]
                                  set dataset name
  --batch_size INTEGER            the number of batch size.
  --load_params                   If set True, parameters are loaded and args
                                  below are ingored. To load params, model
                                  tuning should be executed previously.

  --lr FLOAT                      learning rate for the central model.
  --clientlr FLOAT                learning rate for client models. ignored
                                  when mode is normal

  --gnn_type [gcn|gin]            support gcn, gin.
  --ratio TEXT                    set ratio of the biggest dataset in total
                                  datasize. Other datasets are equally
                                  divided. (0, 1)

  --help                          Show this message and exit.
```

## tensorboard

```shell
$ tensorboard --logdir=logs
```