#!/bin/bash -l

path2ds="/home/rsk3900/Datasets/"
epochs=2
workers=12
lr=0.0005

spack env activate riteyes4

model="ritnet_v1"
declare -a curObj_list=("NVGaze" "PupilNet" "OpenEDS" "Fuhl" "riteyes_general" "LPW")
declare -a batchsize_list=("16" "16" "16" "16" "16" "16")
declare -a selfCorr_list=("0" "1")
declare -a disentangle_list=("0")

test_mode="baseline"
for i in "${!curObj_list[@]}"
do
    for selfCorr in "${selfCorr_list[@]}"
    do
        for disentangle in "${disentangle_list[@]}"
        do
            batchsize=${batchsize_list[i]}
            baseJobName="RC_e2e_system_test_${model}_${curObj_list[i]}_${selfCorr}_${disentangle}"
            str="#!/bin/bash\npython3 train.py --path2data=${path2ds} --expname=${baseJobName} --test_mode=${test_mode} "
            str+="--curObj=${curObj_list[i]} --batchsize=${batchsize} --workers=${workers} --prec=32 --epochs=${epochs} "
            str+="--disp=0 --overfit=20 --lr=${lr} --selfCorr=${selfCorr} --disentangle=${disentangle} --model=${model} "
            echo $str
            echo -e $str > command.lock
            sbatch -J ${baseJobName} -o "rc_log/system_test/${baseJobName}.o" -e "rc_log/system_test/${baseJobName}.e" --mem=16G --cpus-per-task=9 -p debug -A riteyes --gres=gpu:p4:2 -t 0-5:0:0 command.lock
        done
    done
done