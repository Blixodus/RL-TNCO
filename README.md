# RL-TNCO (Fork)

This repository is a fork of [RL-TNCO](https://github.com/NVlabs/RL-TNCO). We recommend reading their `README.md` file in addition to this one for a comprehensive understanding. This work applies the reinforcement learning approach to solving the tensor network contraction ordering problem, specifically for tensor-train scalar products. It is integrated as a submodule of the [TT-ScalProdOpt](https://github.com/Blixodus/TT-ScalProdOpt) tensor-train scalar product contraction ordering project. This repository is public primarily for data replication purposes. However, it may also serve as a resource for those interested in understanding and utilizing RL-TNCO in general.

## Conda environment

To use RL-TNCO, we recommend setting up the provided [Conda](https://www.anaconda.com/) environment. While the original RL-TNCO repository includes a `Dockerfile`, it seems non functional and has been removed from this fork. To set up the Conda environment, run:
```bash
# Create and set up the environment
conda env create -n rl-tnco --file environment.yml
# Install additional dependencies which require specific PyTorch and CUDA versions
conda activate rl-tnco
pip install pyg-lib==0.1.0+pt113cu117 torch-cluster==1.6.0+pt113cu117 torch-scatter==2.1.0+pt113cu117 torch-sparse==0.6.16+pt113cu117 torch-spline-conv==1.2.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html
```

RL-TNCO integrates with [Weights & Biases](https://wandb.ai/), but we recommend disabling it unless necessary to ensure smooth execution. To disable logging, use:
```bash
wandb offline
```

## Reproducing results from the paper

### Pre-trained models

The `models` directory (submodule) contains pre-trained models for $x^Ty$ and $x^TAy$ types of tensor-train scalar products. To use them, update the `config.py` file at **line 27** as follows:
```python
'pretrained_model' : 'models/[type]/epoch_32.model'
```

### Creating training and evaluation datasets

Detailed steps for generating training and evaluation datasets are provided in the parent repository. For the following instructions, we assume that datasets have already been created. Note the differences between dataset formats:
- **Training datasets**: A single file containing multiple training tensor networks.
- **Evaluation datasets**: Multiple files, each containing a single tensor network.

### Training the model

If you wish to train the model instead of using one of the provided pre-trained models, use the `main.py` script. Before running it, update the `config.py` file as required:
```python
# Set these values based on the network type
# x'y-type: 2×length nodes, 3×length edges
# x'Ay-type: 3×length nodes, 5×length edges
n_nodes = [number of nodes of the largest network]
n_edges = [number of edges of the largest network]
# Specify training dataset to prevent random file generation
train_files = '[training_file].p'
# No pre-trained model should be used for training
'pretrained_model' : None
```

After training, a model will be saved at `wandb/run-[run_timestamp_and_id]/files/epoch_32.model`. This model can then be used as pre-trained model for benchmarking.

### Running the benchmark

To run benchmarking, ensure that a pre-trained model is set in `config.py`, then execute the following command. Set `[input]` to a directory containing evaluation datasets (pickle files) and `[output]` to the directory where results should be saved:
```bash
python benchmarking.py [input] [output]
```

### Using results

Once benchmarking is complete, follow the instructions in the parent repository to generate plots.

**Important note: For specific tensor trains, the RL-TNCO method may fail to return a result or only provide a partial result. The cause of this issue remains unknown, however, it is mitigated somewhat by having each benchmarking file contain only a single tensor network**
