DATA_DIR="/media/rakshit/Monster"

PATH_SETS=DATA_DIR+"/Datasets"

echo "Do you want to visualize extraction figures? Warning: This will create a lot of figures"
read plotCond
if ["$plotCond" == "yes"]
then
    noDisp=0
    echo "Plotting figures"
else
    noDisp=1
    echo "Silent process"
fi
# Generate pupil and iris fits offline
# Extract images and generate master key
python3 dataset_generation/ExtractFuthl.py --noDisp=${noDisp} --DATA_DIR=${PATH_SETS}
python3 dataset_generation/ExtractLPW.py --noDisp=${noDisp} --DATA_DIR=${PATH_SETS}
python3 dataset_generation/ExtractPupilNet.py --noDisp=${noDisp} --DATA_DIR=${PATH_SETS}
python3 dataset_generation/ExtractOpenEDS_seg.py --noDisp=${noDisp} --DATA_DIR=${PATH_SETS}
python3 dataset_generation/ExtractNVGaze.py --noDisp=${noDisp} --DATA_DIR=${PATH_SETS}

#python3 dataset_generation/ExtractUnityEyes.py --noDisp=${noDisp} --DATA_DIR=${DATA_DIR}
#python3 dataset_generation/ExtractSantini.py --noDisp=${noDisp} --DATA_DIR=${DATA_DIR}
