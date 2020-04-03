#!/bin/bash
cond=0
selfCorr=1
disentangle=1
expname="Local_e2e_${cond}_${selfCorr}_${disentangle}"
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --curObj=$cond --disp=0 --expname=expname --overfit=0 --epochs=100 --lr=5e-4 --workers=6 --batchsize=12 --selfCorr=$selfCorr --disentangle=$disentangle
