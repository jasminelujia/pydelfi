#!/bin/bash
#SBATCH -J cosmic_shear_map
#SBATCH -t 02-00:00:00
#SBATCH --exclusive        
#SBATCH -o OUTPUT.o%j             
#SBATCH -e OUTPUT.e%j                 
#SBATCH --mail-user=charnock@iap.fr
#SBATCH --mail-type=ALL  
#SBATCH -p gpu --gres=gpu:1 -n 1

module purge
module load slurm
module load gcc/8.2.0 cuda/9.0.176 cudnn/v7.0-cuda-9.0 python3/3.6.2 openmpi/1.10.6-hfi

cd /mnt/home/tcharnock/delfi/
OMP_NUM_THREADS=1 mpirun -np 1 python3 cosmic_shear_map.py
