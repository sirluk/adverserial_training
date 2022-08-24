# Adverserial Training for Bias Mitigation

This code implements adverserial training for transformer models

## Dataset

### Download Data

The dataset used is based on the following paper:

```
Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, Adam Kalai. Bias in Bios: A Case Study of Semantic Representation Bias in a High Stakes Setting. Proceedings of FAT*, 2019
```

It can be reproduced according to the instructions in this repository: https://github.com/Microsoft/biosbias

### Dataset Preperation

To prepera the dataset for training the following jupyter notebook has to be executed: `prepare_data.ipynb`

Here the text is preprocessed, train, validation and test splits are created and a vocabulary is created.

## Installation

To run the code make sure conda is installed and then run

```bash
conda env create -f environment.yml
```

Then activate the environment by running

```bash
conda activate adv_training
```

## Architecture

The project structure looks as follows

📦adverserial_training \
 ┣ 📂src \
 ┃ ┣ 📂models (directory which contains all model classes)\
 ┃ ┃ ┣ 📜model_adv.py (baseline model for adverserial training) \
 ┃ ┃ ┣ 📜model_base.py (contains base class with methods that are used by all models) \
 ┃ ┃ ┣ 📜model_heads.py (classifier and adverserial head classes) \
 ┃ ┃ ┣ 📜model_task.py (baseline model for task training) \
 ┃ ┣ 📜adv_attack.py (contains function to run adverserial attack) \
 ┃ ┣ 📜data_handler.py \
 ┃ ┣ 📜metrics.py \
 ┃ ┣ 📜training_logger.py \
 ┃ ┗ 📜utils.py \
 ┣ 📜cfg.yml (hyperparameters)\
 ┣ 📜environment.yml (conda environment config)\
 ┣ 📜main.py (main file to run experiments with)\
 ┣ 📜prepare_data.ipynb (script to prepare BIOS dataset so it can be imported by the data handler)\
 ┗ 📜readme.md

\* Weight parametrizations are implemented as modules and use pytorch parametrizations functionality [LINK](https://pytorch.org/tutorials/intermediate/parametrizations.html)

## cfg.yml

contains hyperparameter configuration

* data_config \
filepaths to data files
* model_config \
name of pretrained model and batch_size to use
* train_config_baseline \
hyperparameters for model (model_adv.py and model_task.py)
* adv_attack
hyperparameters for adverserial attack

## Usage

```bash
python3 main.py
```

Optional arguments with example inputs

* --gpu_id=0 \
Which gpu to run experiment on (defaults to 0)
* --cpu \
Run experiment on cpu
* --adv \
Run adverserial training
* --seed=0 \
Random seed for pytorch (defaults to 0)
* --debug \
Run in debug mode (limited dataset, only one epoch)
* --no_adv_attack \
Set if you do not want to run adverserial attack after training
