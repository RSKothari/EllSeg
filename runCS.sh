#!/bin/bash
# keep resume on because jobs keep crashing on CS server
cond=0
selfCorr=1
workers=2
expname1="CS_e2e_${cond}_${selfCorr}_0"
expname2="CS_e2e_${cond}_${selfCorr}_1"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 train.py --curObj=$cond \
--disp=0 --disentangle=0 --expname=$expname1 --overfit=0 --epochs=100 \
--lr=5e-4 --workers=$workers --batchsize=40 --selfCorr=${selfCorr} \
--path2data=/home/group2/research/cgaplab/ --resume=1 &
CUDA_VISIBLE_DEVICES=5,6,7,8,9 python3 train.py --curObj=$cond \
--disp=0 --disentangle=1 --expname=$expname2 --overfit=0 --epochs=100 \
--lr=5e-4 --workers=$workers --batchsize=40 --selfCorr=${selfCorr} 
-path2data=/home/group2/research/cgaplab/ --resume=1 &