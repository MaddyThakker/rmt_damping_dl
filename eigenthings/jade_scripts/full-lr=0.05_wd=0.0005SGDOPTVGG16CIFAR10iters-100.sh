#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=vggruns
#SBATCH --partition=small
#SBATCH --gres=gpu:1
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00000.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00000 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00000
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00025.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00025 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00025
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00050.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00050 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00050
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00075.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00075 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00075
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00100.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00100 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00100
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00125.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00125 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00125
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00150.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00150 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00150
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00175.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00175 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00175
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00200.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00200 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00200
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00225.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00225 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00225
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00250.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00250 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00250
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00275.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00275 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00275
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005checkpoint-00300.pt --basis_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00300 --spectrum_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD/lr=0.05_wd=0.0005hessian-100-00300
