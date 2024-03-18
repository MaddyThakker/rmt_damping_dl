#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=sgdVGG16

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

# Weight decay = 0
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/VGG16/sgd/sgd_wd0_lr_5e-4_slr_25e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.05 --wd 0 --epochs 300 --save_freq=5  

# Weight decay = 1e-5
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/VGG16/sgd/sgd_wd1e-5_lr_5e-4_slr_25e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.05 --wd 1e-5 --epochs 300 --save_freq=5 

# Weight decay = 5e-5
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/VGG16/sgd/sgd_wd5e-5_lr_5e-4_slr_25e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.05 --wd 5e-5 --epochs 300 --save_freq=5 

# Weight decay = 1e-4
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/VGG16/sgd/sgd_wd1e-4_lr_5e-4_slr_25e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.05 --wd 1e-4 --epochs 300 --save_freq=5 

# Weight decay = 5e-4
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/VGG16/sgd/sgd_wd5e-4_lr_5e-4_slr_25e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.05 --wd 5e-4 --epochs 300 --save_freq=5

# Weight decay = 1e-3
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/VGG16/sgd/sgd_wd1e-3_lr_5e-4_slr_25e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.05 --wd 1e-3 --epochs 300 --save_freq=5 