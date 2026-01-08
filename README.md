# Block-wise Noisy Fine-Tuning

This repository contains code for Sequential Subspace Noise Injection Prevents Accuracy
Collapse in Certified Unlearning.

The code supports training, retraining, and NFT and Blockwise-NFT unlearning experiments on MNIST and CIFAR-10.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Experiment hyperparameters are read from YAML configs: ```experiments/<experiment_type>/<modelname>.yaml```.

## Quick start

All experiments are launched via the main entry point ```main.py```.


### Train a base model (full dataset)

```bash
python main.py --mode train_base --dataset cifar10 --training
```

### Retrain from scratch (retain set only)

```bash
python main.py --mode retrain --dataset cifar10 --training
```

### Noisy Fine-Tuning (NFT)

```bash
python main.py --mode unlearning_base --dataset cifar10 --training \\
  --trained_full_state_path path/to/full_model.pt
```

### Blockwise Noisy Fine-Tuning (our method)

```bash
python main.py --mode blocks --dataset cifar10 --training \\
  --trained_full_state_path path/to/full_model.pt \\
  --blocks_mode sequential \\
  --blocks_split_type random
```


