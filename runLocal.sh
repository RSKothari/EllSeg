#!/bin/bash
cond=0
selfCorr=1
disentangle=0
expname="Local_e2e_${cond}_${selfCorr}_${disentangle}"
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --resume=1 --curObj=$cond --disp=1 --expname=$expname --overfit=0 --epochs=100 --lr=5e-4 --workers=5 --batchsize=10 --selfCorr=$selfCorr --disentangle=$disentangle
