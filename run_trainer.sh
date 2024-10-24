#!/bin/sh
#SBATCH --job-name=dquartic_training         # Job name
#SBATCH --account=def-hroest                  # Account name
#SBATCH --time=5-00:00:00                      # Time limit (5 days)
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks=1                             # Number of tasks (usually 1 for GPU jobs)
#SBATCH --cpus-per-task=4                      # Number of CPU cores per task
#SBATCH --mem=16G                              # Memory per node
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --output=dquartic_train_%j.log        # Output file
#SBATCH --error=dquartic_train_%j.err         # Error file

# source code/py310/bin/activate

# Usage: dquartic train [OPTIONS]

#   Train a DDIM model on the DIAMS dataset.

# Options:
#   --epochs INTEGER           Number of epochs to train
#   --warmup-epochs INTEGER    Number of warmup epochs for learning rate
#                              scheduler
#   --batch-size INTEGER       Batch size for training
#   --learning-rate FLOAT      Learning rate for optimizer
#   --hidden-dim INTEGER       Hidden dimension for the model
#   --num-heads INTEGER        Number of attention heads
#   --num-layers INTEGER       Number of transformer layers
#   --normalize TEXT           Normalization method. (None, minmax)
#   --ms2-data-path TEXT       Path to MS2 data
#   --ms1-data-path TEXT       Path to MS1 data
#   --checkpoint-path TEXT     Path to save the best model
#   --use-wandb                Enable Weights & Biases logging
#   --threads INTEGER          Number of threads for data loading
#   --wandb-project TEXT       Weigths & Biases project name
#   --wandb-name TEXT          Weigths & Biases name. If None, Wandb will
#                              generate a random name
#   --wandb-id TEXT            Weigths & Biases run ID
#   --wandb-resume TEXT        Weigths & Biases resume ID if run crashed or
#                              stopped early. Allowed values are None, allow,
#                              must, never
#   --wandb-architecture TEXT  Weigths & Biases model architecture name
#   --wandb-dataset TEXT       Weigths & Biases dataset name
#   --wandb-mode TEXT          Weigths & Biases mode. (offline, online). Default
#                              is offline, which means metrics are only saved
#                              locally. You will need to run wandb sync to
#                              upload the metrics to the cloud.
#   --help                     Show this message and exit.


epochs=1000
warmup_epochs=5
batch_size=1
learning_rate=0.00001
hidden_dim=1024
num_heads=8
num_layers=8
normalize='minmax'
ms2_data_path='data/ms2_data_cat_int32.npy'
ms1_data_path='data/ms1_data_int32.npy'
checkpoint_path="best_model.pth"
use_wandb=True
threads=4
wandb_name='experiment_1'

dquartic train --epochs $epochs --warmup-epochs $warmup_epochs --batch-size $batch_size --learning-rate $learning_rate --hidden-dim $hidden_dim --num-heads $num_heads --num-layers $num_layers --normalize $normalize --ms2-data-path $ms2_data_path --ms1-data-path $ms1_data_path --threads $threads --checkpoint-path $checkpoint_path --use-wandb --wandb-name $wandb_name