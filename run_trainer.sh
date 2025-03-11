#!/bin/bash
#SBATCH --job-name=dquartic_training         # Job name
#SBATCH --account=def-hroest                  # Account name
#SBATCH --time=7-00:00:00                      # Time limit (5 days)
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks=1                             # Number of tasks (usually 1 for GPU jobs)
#SBATCH --cpus-per-task=4                      # Number of CPU cores per task
#SBATCH --mem=16G                              # Memory per node
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --output=dquartic_train_%j.log        # Output file
#SBATCH --error=dquartic_train_%j.err         # Error file

# source code/py310/bin/activate
source py310/bin/activate

# On Beluga and Narval, we have to run wandb in offline mode, as the compute nodes do not have access to the internet and cannot sync the content. We would need to sync after the job has been completed with wandb sync

wandb offline

config_path="dquartic_train_config_narval.json"
wandb_name='20241116_lr_1eneg4_xo'
# wandb_name='20241026_unet1d_ms2_loss_pred_tgt' 
checkpoint_path="experiments/$wandb_name/best_model.pth"

# Update config file for wandb_name and checkpoint path
jq --arg wandb_name "$wandb_name" '.wandb.wandb_name = $wandb_name' "$config_path" > tmp.json && mv tmp.json "$config_path"
jq --arg wandb_name "$wandb_name" '.wandb.wandb_id = $wandb_name' "$config_path" > tmp.json && mv tmp.json "$config_path"
jq --arg checkpoint_path "$checkpoint_path" '.model.checkpoint_path = $checkpoint_path' "$config_path" > tmp.json && mv tmp.json "$config_path"

mkdir -p "experiments/$wandb_name"

dquartic train $config_path
