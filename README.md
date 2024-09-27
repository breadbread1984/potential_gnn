# Introduction

this project predict potential with GNN

# Usage

## Install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## Download dataset

```shell
bash download_datasets.sh
```

## Train Model

```shell
python3 train.py --trainset <path/to/trainset> --evalset <evalset>
```

## Evaluate Model

```shell
python3 evaluate.py --evalset <evalset> --ckpt <path/to/model.pt>
```
