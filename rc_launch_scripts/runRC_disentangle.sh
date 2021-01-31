#!/bin/bash -l

path2ds="/home/rsk3900/Datasets/"
epochs=40
workers=12
lr=0.0005

spack env activate riteyes4

#model="ritnet_v1"
declare -a model_list=('ritnet_v1' 'ritnet_v2' 'ritnet_v3' 'ritnet_v4' 'ritnet_v5' 'ritnet_v6')
declare -a curObj_list=("NVGaze" "PupilNet" "OpenEDS" "Fuhl" "riteyes_general" "LPW")
declare -a batchsize_list=("48" "48" "48" "48" "48" "48")
declare -a selfCorr_list=("0")
declare -a disentangle_list=("1")

declare -a tests=("leaveoneout")

for model in "${model_list[@]}"
do
    for test_mode in "${tests[@]}"
    do
        for i in "${!curObj_list[@]}"
        do
            for selfCorr in "${selfCorr_list[@]}"
            do
                for disentangle in "${disentangle_list[@]}"
                do
                    batchsize=${batchsize_list[i]}
                    baseJobName="RC_e2e_${test_mode}_${model}_${curObj_list[i]}_${selfCorr}_${disentangle}"
                    str="#!/bin/bash\npython3 train.py --path2data=${path2ds} --expname=${baseJobName} --test_mode=${test_mode} "
                    str+="--curObj=${curObj_list[i]} --batchsize=${batchsize} --workers=${workers} --prec=32 --epochs=${epochs} "
                    str+="--disp=0 --overfit=0 --lr=${lr} --selfCorr=${selfCorr} --disentangle=${disentangle} --model=${model}"
                    echo $str
                    echo -e $str > command.lock
                    sbatch -J ${baseJobName} -o "rc_log/${test_mode}/${baseJobName}.o" -e "rc_log/${test_mode}/${baseJobName}.e" --mem=16G --cpus-per-task=9 -p tier3 -A riteyes --gres=gpu:v100:1 -t 2-0:0:0 command.lock
                done
            done
        done
    done
done
