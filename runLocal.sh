#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --curObj=0 --disp=0 --expname=cond_0 --overfit=0 --epochs=1000 --lr=5e-4 --workers=6 --batchsize=12 --selfCorr=0 --disentangle=1 --resume=1 --loadfile=/home/rakshit/Documents/Python_Scripts/GIW_e2e/logs/ritnet/cond_0/weights/ritnet_15.pkl
