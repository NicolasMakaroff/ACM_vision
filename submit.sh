#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -c 2
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=0-15:00:00
#SBATCH --job-name="CCV_GPU"
#SBATCH --output=./output/%x.%J.out
#SBATCH --error=./error/%x.%J.out
#SBATCH --mail-user=nicolas.makaroff@dauphine.psl.eu
#SBATCH --mail-type=BEGIN,END,FAIL

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_IB_DISABLE=1

# on your cluster you might need these:
# set the network interface
#export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python3 train.py --epochs 50 --image_size 256 
