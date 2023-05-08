# GGNN with ADA-NNS
This repository is GGNN modified to run both single query and batch processing.

Please refer to original [readme](https://github.com/SNU-ARC/ggnn/blob/release_0.5/README.md).

# Prerequisites
Our prerequisites align with original GGNN [repository](https://github.com/SNU-ARC/ggnn). 

Requirements:
* CUDA (>10.2)
* libgflags-dev (`sudo apt install libgflags-dev`)

# Usage
We provide script which builds graph, run queries and parses the result. All results can be found on folders **result/ggnn_single_batch**, **result/ggnn_whole_batch** in csv format.

```shell
$ git clone https://github.com/SNU-ARC/ggnn
$ cd ggnn/
$ git checkout ADA-NNS
$ sudo ln -s YOUR_DATASETS_FOLDER datasets
$ ./script/run.sh
$ mv result/single_* result/single
$ mv result/whole_* result/whole
$ python ./script/parse.py
```
