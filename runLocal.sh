#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3.7 train.py --curObj=0 --disp=0 --expname=dev_center --overfit=0 --epochs=100 --lr=5e-4 --workers=8 --batchsize=16
