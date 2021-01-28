#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:08:51 2020

@author: rakshit

The purpose of this script is to verify if train/test objects are working as
intended. This function will iterate over H5 files and display groundtruth
annotations (whichever are present)
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from helperfunctions import mypause
from torch.utils.data import DataLoader
from utils import generateImageGrid, points_to_heatmap
from CurriculumLib import readArchives, listDatasets, generate_fileList
from CurriculumLib import selDataset, selSubset, DataLoader_riteyes


if __name__=='__main__':
    path2data = '/media/rakshit/tank/Dataset'
    path2h5 = os.path.join(path2data, 'All')
    path2arc_keys = os.path.join(path2data, 'MasterKey')

    # NV, Fuhl, PN, LPW, riteyes_general, OpenEDS
    path_train_test_object = os.path.join('curObjects', 'baseline', 'cond_pretrained.pkl')
    trainObj, validObj, _ = pickle.load(open(path_train_test_object, 'rb'))
    trainObj.path2data = path2h5

    # Train loader
    trainLoader = DataLoader(trainObj,
                             batch_size=32,
                             shuffle=True,
                             num_workers=8,
                             drop_last=True)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    totTime = []
    startTime = time.time()
    for bt, data in enumerate(trainLoader):
        img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = data

        # Display annotated image
        dispI = generateImageGrid(img.squeeze().numpy(),
                                  labels.numpy(),
                                  elNorm.detach().cpu().numpy().reshape(-1, 2, 5),
                                  pupil_center.numpy(),
                                  cond.numpy(),
                                  override=True,
                                  heatmaps=False)

        dT = time.time() - startTime
        totTime.append(dT)
        print('Batch: {}. Time: {}'.format(bt, dT))
        startTime = time.time()

        if bt == 0:
            h_ims = axs.imshow(0.5*dispI.permute(1, 2, 0)+0.5, cmap='gray')
            plt.show(block=False)
            plt.pause(0.01)
        else:
            h_ims.set_data(0.5*dispI.permute(1, 2, 0)+0.5)
            mypause(0.01)
    print('Time for 1 epoch: {}'.format(np.sum(totTime)))
