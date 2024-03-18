#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=SGDSpectrum

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=xingchen.wan@outlook.com


# run the application
module load python3/anaconda
source activate diegorubin
python3 spectrum.py  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model PreResNet110 --ckpt out/PreResNet110/SGD_noschedule/SGD_wd5e-4/checkpoint-00000.pt --basis_path  out/spectrum/PreResNet110/SGD_noschedule/00300.npz --iters 100
python3 spectrum.py --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model PreResNet110 --ckpt out/PreResNet110/SGD_noschedule/SGD_wd5e-4/checkpoint-00000.pt --basis_path  out/spectrum/PreResNet110/SGD_noschedule/00300-batch.npz --iters 100 --num_samples=128