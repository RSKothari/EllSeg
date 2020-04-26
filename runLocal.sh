#!/bin/bash

trainSet='Fuhl'
testSet='PupilNet' 
selfCorr=0
disentangle=0
cond='Cond_train_${trainSet}_test_${testSet}'
expname="Local_e2e_${cond}_${selfCorr}_${disentangle}"
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --resume=1 --curObj='curObjects/baseline/${cond}' --disp=1 --expname=$expname --overfit=40 --epochs=100 --lr=5e-4 --workers=5 --batchsize=10 --selfCorr=$selfCorr --disentangle=$disentangle
