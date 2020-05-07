#!/bin/bash

trainSet='NVGaze'
selfCorr=0
disentangle=0
expname="Local_e2e_${trainSet}_${selfCorr}_${disentangle}"
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --resume=0 --curObj=${trainSet} --disp=1 --expname=$expname --overfit=0 --epochs=50 --lr=5e-5 --workers=6 --batchsize=12 --selfCorr=$selfCorr --disentangle=$disentangle
