#!/bin/bash -l

path2ds="/home/rsk3900/Datasets/"
epochs=100
workers=12
lr=0.0005

spack env activate riteyes4

model="ritnet_v1"
declare -a curObj_list=("NVGaze" "PupilNet" "OpenEDS" "Fuhl" "riteyes-general" "LPW")
declare -a batchsize_list=("48" "60" "48" "60" "48" "60")
declare -a selfCorr_list=("0")
declare -a disentangle_list=("0")

for i in "${!curObj_list[@]}"
do
    for selfCorr in "${selfCorr_list[@]}"
    do
        for disentangle in "${disentangle_list[@]}"
        do
            batchsize=${batchsize_list[i]}
            baseJobName="RC_e2e_${model}_${curObj_list[i]}_${selfCorr}_${disentangle}"
            str="#!/bin/bash\npython3 train.py --path2data=${path2ds} --expname=${baseJobName} "
            str+="--curObj=${curObj_list[i]} --batchsize=${batchsize} --workers=${workers} --prec=32 --epochs=${epochs} "
            str+="--disp=0 --overfit=0 --lr=${lr} --selfCorr=${selfCorr} --disentangle=${disentangle} --model=${model}"
            echo $str
            echo -e $str > command.lock
            sbatch -J ${baseJobName} -o "rc_log/baseline/${baseJobName}.o" -e "rc_log/baseline/${baseJobName}.e" --mem=16G --cpus-per-task=9 -p tier3 -A riteyes --gres=gpu:v100:1 -t 2-0:0:0 command.lock
        done
    done
done
