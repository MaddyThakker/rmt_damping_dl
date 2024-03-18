#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=KFAC

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
source activate curvature

# KFAC
# Weight decay = 0
python3 run_KFAC.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/KFAC/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.3 --wd 1e-4 --epochs 150 --save_freq=25
