# RL-TNCO (fork)

This is a fork of the [RL-TNCO](https://github.com/NVlabs/RL-TNCO) project; we recommend reading their `README.md` file in addition to this one. The purpose is to apply the reinforcement learning method for finding contraction orders in tensor networks onto tensor-train scalar products as a submodule of [the OptiTenseurs tensor-train scalar product contraction ordering project](https://github.com/Blixodus/OptiTenseurs). This repository is public mainly for data replication purposes; however, some of this information might be useful to users who wish to understand how to use RL-TNCO in general.

## Conda environment

To use RL-TNCO, we recommend using the provided [conda](https://www.anaconda.com/) environment. The original RL-TNCO repository also contains a `Dockerfile`, which unfortunately does not work and has been removed from this fork. To create the conda environment, use the following commands:
```
# Pre-setup environment
conda env create -n rl-tnco --file environment.yml
# Install additional packages dependent on specific PyTorch and CUDA versions and not available by default (requires activation of the environment)
conda activate rl-tnco
pip install pyg-lib==0.1.0+pt113cu117 torch-cluster==1.6.0+pt113cu117 torch-scatter==2.1.0+pt113cu117 torch-sparse==0.6.16+pt113cu117 torch-spline-conv==1.2.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html
```

The RL-TNCO project uses an [online tool](https://wandb.ai/); however, we recommend turning it off unless necessary to ensure the code runs well in all cases. For this, you can for example use the following command:
```
wandb offline
```

## Provided files for tensor-train scalar product

The `datasets` directory contains the files that were used for model training and evaluation in the [tensor-train scalar product contraction ordering publication (link to come)](). Notably,
- `train/scalar_product_2D_dataset_num_eqs_900_num_node_100_mean_conn_3.p` can be used to train on $x^Ty$-type scalar products,
- `train/scalar_product_3D_dataset_num_eqs_450_num_node_100_mean_conn_3.p` can be used to train on $x^TAy$-type scalar products,
- `eval/xy/` directory contains the $x^Ty$-type tensor trains used to evaluate the model,
- `eval/xAy/` directory contains the $x^TAy$-type tensor trains used to evaluate the model,
- `eval/real-tt/` directory contains the real-life tensor trains used to evaluate the model.

The `models` directory contains pre-trained models for $x^Ty$ and $x^TAy$ types of tensor-train scalar products. To use them, modify the `config.py` file at line 27 to be as follows:
```
'pretrained_model' : 'models/[type]/epoch_32.model'
```

## Reproducing data from the paper

You can run the `benchmarking.py` script after setting a model, as described in the previous section. Run the command below, setting input to one of the directories listed above and output to any directory where you want results to be written. After obtaining results, you can follow instructions from [OptiTenseurs](https://github.com/Blixodus/OptiTenseurs) to create plots.
```
python benchmarking.py [input] [output]
```

**Important note: For some tensor trains, the RL-TNCO method might not return a result or only a partial result. The reason for this is currently unknown to us.**

## Training the model

If you wish to train a model, run the `main.py` script. Simply modify the config.py file as required:
```
# Make these values match reality (note: xy type network has 2 x length nodes and 3 x length edges, xAy type network has 3 x length nodes and 5 x length edges)
n_nodes = [number of nodes of the largest network]
n_edges = [number of edges of the largest network]
# Point to training files to avoid scripts creating random training files
train_files = 'datasets/scalar_product_[type].p'
# No pre-trained model for training
'pretrained_model' : None
```

You can use one of the previously mentioned training files or generate your own with tools from [OptiTenseurs](https://github.com/Blixodus/OptiTenseurs).
