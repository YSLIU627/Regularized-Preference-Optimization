# Codebase for Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer
This codebase is adapted from the [alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main).
## Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.1.2` - the precise version is important for reproducibility! Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies as follows:

```shell
cd alignment-handbook/
python -m pip install -e .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn==2.3.6 --no-build-isolation
```

> **Note**
> If your machine has less than 96GB of RAM and many CPU cores, reduce the `MAX_JOBS` arguments, e.g. `MAX_JOBS=4 pip install flash-attn==2.3.6 --no-build-isolation`

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

You can now check out the `scripts` and `recipes` directories for instructions on how to train some models 🪁!

## Project structure

```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
├── README.md                   <- The top-level README for developers using this project
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── scripts                     <- Scripts to train and evaluate chat models
├── setup.cfg                   <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `alignment` can be imported
├── src                         <- Source code for use in this project
```

## Run training on beta series models
We use 8 NVIDIA A100 GPUs for the training.
### Run DPO (beta)

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes=8  scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml
```

### Run RPO (beta) with eta = 0.005

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes=8  scripts/run_rpo.py  recipes/zephyr-7b-beta/rpo/config_full.yaml  
```
You can modify the choice of eta for RPO in `recipes/zephyr-7b-beta/rpo/config_full.yaml` .

## Run training on gemma series models
We use 8 NVIDIA A6000 GPUs for the training.

### Run DPO (gemma)

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes=8  scripts/run_dpo.py recipes/zephyr-7b-gemma/dpo/config_full.yaml
```

### Run RPO (gemma) with eta = 0.005

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes=8  scripts/run_rpo.py  recipes/zephyr-7b-gemma/rpo/config_full.yaml  
```
You can modify the choice of eta for RPO in `recipes/zephyr-7b-gemma/rpo/config_full.yaml` .
