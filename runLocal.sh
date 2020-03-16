#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --curObj=0 --disp=0 --expname=dev_overfit --overfit=20 --epochs=1000 --lr=5e-4 --workers=6 --batchsize=12
