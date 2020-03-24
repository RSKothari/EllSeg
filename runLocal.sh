#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --curObj=1 --disp=0 --expname=cond_1 --overfit=0 --epochs=1000 --lr=5e-4 --workers=6 --batchsize=12 --selfCorr=0
