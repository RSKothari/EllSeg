#!/bin/bash -l
#SBATCH --job-name=GIW_e2e
#SBATCH --output=rc_log/GIW_e2e.out
#SBATCH --error=rc_error/GIW_e2e.err

#SBATCH --mail-user rsk3900@rit.edu
#SBATCH --mail-type=ALL

#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH -t 2-0:0:0
#SBATCH -p tier3 -A riteyes
#SBATCH --cpus-per-task=9

path2ds="/home/rsk3900/Datasets/"
model="ritnet"
expname="curCond_0"

# Load necessary modules
spack load /7qmaaiw # Load OpenCV
spack load /jthz32l # Load pytorch by hash
spack load /dtlfq7l # Load torchvision by hash
spack load /fvki7dt # Load scipy
spack load /rso7arf # Load matplotlib
spack load /me57ozl # Load image manipulation library
spack load /bblye5g # Load sklearn for metrics
spack load /be2kd5v # Load tensorboardx

python3 train.py --disp=0 --path2ds=$path2ds --model=$fName --expname=$expname --batchsize=24 --workers=8 --prec=32 --epochs=50
