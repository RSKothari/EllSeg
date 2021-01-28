# EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking

# Abstract
Ellipse fitting, an essential component in pupil or iris tracking based video oculography, is performed on previously segmented eye parts generated using various computer vision techniques. Several factors, such as occlusions due to eyelid shape, camera position or eyelashes, frequently break ellipse fitting algorithms that rely on well-defined pupil or iris edge segments. In this work, we propose training a convolutional neural network to directly segment entire elliptical structures and demonstrate that such a framework is robust to occlusions and offers superior pupil and iris tracking performance (at least 10$\%$ and 24$\%$ increase in pupil and iris center detection rate respectively within a two-pixel error margin) compared to using standard eye parts segmentation for multiple publicly available synthetic segmentation datasets.

# Pretrained models
EllSeg is a framework which can easily be replicated on any encoder-decoder architecture. To facilitate our work, we create a custom network nicknamed DenseElNet for which we provide trained models as reported in the [paper](https://arxiv.org/abs/2007.09600).

Trained on:
- OpenEDS
- NVGaze
- RITEyes
- LPW*
- Fuhl*
- PupilNet*
- All datasets (best for deploying)

To ensure stable training, starred models * were initialized with weights from a network pretrained on OpenEDS+NVGaze+RITEyes for 2 epochs. To replicate results on starred * sets,  please initialize with the following pretrained weights.

# Try it out on your eye videos!
For quick inference on your own eye videos, please use `evaluate_ellseg.py` as `python evaluate_ellseg --path_dir=${PATH_EYE_VIDEOS}`. This scripts expects eye videos in the following folder hierarchy (at most two eye videos under one experiment - this follows the PupilLabs format).
- `${path_eye_videos}`
	- exp_name_0 (can be whatever)
		- eye0.mp4
		- eye1.mp4
	- exp_name_1
		- eye0.mp4

# Pupil Labs integration
Coming soon! 

# Downloading datasets

Since we do not have access to publish or share other datasets, please download the following datasets and place them in `${DATA_DIR}/Datasets`. Links updated on 27/01/2021.
1. [ElSe + ExCuse](https://atreus.informatik.uni-tuebingen.de/seafile/d/8e2ab8c3fdd444e1a135/?p=%2Fdatasets-head-mounted&mode=list)
2. [PupilNet](https://atreus.informatik.uni-tuebingen.de/seafile/d/8e2ab8c3fdd444e1a135/?p=%2Fdatasets-head-mounted&mode=list)
3. [LPW](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw/)
4.  [NVGaze](https://sites.google.com/nvidia.com/nvgaze)
5. [RITEyes](https://cs.rit.edu/~cgaplab/RIT-Eyes/)
6. [OpenEDS](https://research.fb.com/programs/openeds-challenge)

## Special data instructions
Combine images from `ElSe` and `ExCuSe` datasets into a common directory names `${DATA_DIR}/Datasets/Fuhl`. This combined dataset will henceforth be referred to as `Fuhl`. To ensure we use the latest pupil centers annotations, please use the files marked with `_corrected` and discard their earlier variants. Rename files as such `data set XXI_corrected.txt` to `data set XXI.txt`. Be sure to unzip image data. The expected hierarchy looks something like this:
- `Datasets`
	- `Fuhl`
		- `data set XXIII.txt`
		- `data set XXIII`
			- `0000000836.png`

Do not unzip the NVGaze dataset, it will consume a lot of wasteful resources and storage space. The code automatically extracts images via zip files. The expected hierarchy looks something like this:

# Citations
If you only use our code base, please cite the following works
EllSeg 
```
@article{kothari2020ellseg,
  title={EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking},
  author={Kothari, Rakshit S and Chaudhary, Aayush K and Bailey, Reynold J and Pelz, Jeff B and Diaz, Gabriel J},
  journal={arXiv preprint arXiv:2007.09600},
  year={2020}
}
```
RITEyes
```
@inproceedings{nair2020rit,
  title={RIT-Eyes: Rendering of near-eye images for eye-tracking applications},
  author={Nair, Nitinraj and Kothari, Rakshit and Chaudhary, Aayush K and Yang, Zhizhuo and Diaz, Gabriel J and Pelz, Jeff B and Bailey, Reynold J},
  booktitle={ACM Symposium on Applied Perception 2020},
  pages={1--9},
  year={2020}
}
```
Please cite and credit individual datasets at their own respective links.

# Questions?
Please email Rakshit Kothari at rsk3900@rit.edu


