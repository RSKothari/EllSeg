#!/bin/bash -l

path2ds="/home/rsk3900/Datasets/"
epochs=100
workers=12
lr=0.0005

spack env activate riteyes4
# Load necessary modules
# spack load /7qmaaiw # Load OpenCV
# spack load /jthz32l # Load pytorch by hash
# spack load /dtlfq7l # Load torchvision by hash
# spack load /fvki7dt # Load scipy
# spack load /rso7arf # Load matplotlib
# spack load /me57ozl # Load image manipulation library
# spack load /bblye5g # Load sklearn for metrics
# spack load /zzdgeg6 # Load tensorboardx (latest)
# spack load /me75cc2 # Load tqdm
# spack load /hlxw2mt # Load h5py with MPI

declare -a curObj_list=("NVGaze" "PupilNet" "OpenEDS" "Fuhl" "riteyes-general" "LPW")
declare -a batchsize_list=("36" "48" "36" "48" "36" "48")
declare -a selfCorr_list=("0")
declare -a disentangle_list=("0")

for i in "${!curObj_list[@]}"
do
    for selfCorr in "${selfCorr_list[@]}"
    do
        for disentangle in "${disentangle_list[@]}"
        do
            batchsize=${batchsize_list[i]}
            baseJobName="RC_e2e_${curObj_list[i]}_${selfCorr}_${disentangle}"
            str="#!/bin/bash\npython3 train.py --path2data=${path2ds} --expname=${baseJobName} "
            str+="--curObj=${curObj_list[i]} --batchsize=${batchsize} --workers=${workers} --prec=32 --epochs=${epochs} "
            str+="--disp=0 --overfit=0 --lr=${lr} --selfCorr=${selfCorr} --disentangle=${disentangle}"
            echo $str
            echo -e $str > command.lock
            sbatch -J ${baseJobName} -o "rc_log/baseline/${baseJobName}.o" -e "rc_log/baseline/${baseJobName}.e" --mem=16G --cpus-per-task=9 -p debug -A riteyes --gres=gpu:p4:2 -t 0-1:0:0 command.lock
        done
    done
done
