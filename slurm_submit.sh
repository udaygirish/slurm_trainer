#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 8                   # Number of cores
#SBATCH --mem=32g          # Memory pool for all cores
#SBATCH -J "MNIST_Train"      # Job name
#SBATCH -p short              # Partition to submit to
#SBATCH -t 02:00:00          # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH -C "V100"            # GPU constraint

# Load required modules
module load python
module load cuda/12.2

# Login to wandb (you'll need to set up WANDB_API_KEY in your environment)
wandb login $WANDB_API_KEY

# Run the training script
python train.py 