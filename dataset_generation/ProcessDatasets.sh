DATA_DIR="/media/rakshit/Monster"

PATH_SETS="${DATA_DIR}/Datasets"

echo "Do you want to visualize extraction figures? Warning: This will create a lot of figures"
read plotCond
if [ ${plotCond} = "yes" ]
then
    noDisp=0
    echo "Plotting figures"
else
    noDisp=1
    echo "Silent process"
fi

mkdir -p "${PATH_SETS}/All"
mkdir -p "${PATH_SETS}/MasterKey"

# Generate pupil and iris fits offline
# Extract images and generate master key
nohup python3 ExtractLPW.py --noDisp=${noDisp} --path2ds=${PATH_SETS} > LPW_status.out & \
nohup python3 ExtractFuthl.py --noDisp=${noDisp} --path2ds=${PATH_SETS} > fuhl_status.out & \
nohup python3 ExtractPupilNet.py --noDisp=${noDisp} --path2ds=${PATH_SETS} > PN_status.out & \
nohup python3 ExtractNVGaze.py --noDisp=${noDisp} --path2ds=${PATH_SETS} > NVGaze_status.out & \
nohup python3 ExtractOpenEDS_seg.py --noDisp=${noDisp} --path2ds=${PATH_SETS} > OpenEDS_status.out & \
nohup python3 ExtractRITEyes_general.py --noDisp=${noDisp} --path2ds=${PATH_SETS} > RITEyes_gen_status.out
