# Codebase for Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer
This codebase is adapted from the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
## Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Then, install the required packages:
```shell
pip install packaging

pip install ninja
MAX_JOBS=4 pip    install flash-attn==2.6.1 --no-build-isolation

pip install -e .
```

For the gemma series models, please run the following command to train the model with RPO:
```shell
bash train_rpo.sh
```
Here, the default eta is 0.2. You can change the eta by modifying the `ETA` in the `train_rpo.sh` file.

And please run the following command to train the model with DPO:
```shell
bash train_dpo.sh
```
