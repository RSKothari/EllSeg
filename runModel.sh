#!/bin/bash -l

#SBATCH --mail-user rsk3900@rit.edu
#SBATCH --mail-type=ALL

#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100
#SBATCH -t 1-0:0:0
#SBATCH -p tier3 -A riteyes
#SBATCH --cpus-per-task=9

#SBATCH --job-name=GIW_e2e_v100
#SBATCH -o rc_log/GIW_e2e_v100.o
#SBATCH -e rc_log/GIW_e2e_V100.e

path2ds="/home/rsk3900/Datasets/"
model="ritnet"
expname="curCond_0_v100"
curObj=0

# Load necessary modules
spack load /7qmaaiw # Load OpenCV
spack load /jthz32l # Load pytorch by hash
spack load /dtlfq7l # Load torchvision by hash
spack load /fvki7dt # Load scipy
spack load /rso7arf # Load matplotlib
spack load /me57ozl # Load image manipulation library
spack load /bblye5g # Load sklearn for metrics
spack load /be2kd5v # Load tensorboardx
spack load /me75cc2 # Load tqdm
spack load /hlxw2mt # Load h5py with MPI

python3 train.py --disp=0 --path2data=$path2ds --model=$model --expname=$expname --curObj=$curObj --batchsize=48 --workers=8 --prec=32 --epochs=50
