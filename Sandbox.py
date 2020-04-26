#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:08:51 2020

@author: rakshit
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import generateImageGrid, points_to_heatmap
from torch.utils.data import DataLoader
from RITEyes_helper.helperfunctions import mypause
from RITEyes_helper.CurriculumLib import readArchives, listDatasets, generate_fileList
from RITEyes_helper.CurriculumLib import selDataset, selSubset, DataLoader_riteyes


if __name__=='__main__':
    path2data = '/media/rakshit/tank/Dataset'
    path2h5 = os.path.join(path2data, 'All')
    path2arc_keys = os.path.join(path2data, 'MasterKey')
    '''
    AllDS = readArchives(path2arc_keys)
    datasets_present, subsets_present = listDatasets(AllDS)
    print('Datasets present -----')
    print(datasets_present)
    print('Subsets present -----')
    print(subsets_present)

    nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 5) for j in range(0, 3)]
    nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 5) for j in range(0, 3)]
    lpw_subs = ['LPW_{}'.format(i+1) for i in range(0, 12)]
    subsets = nv_subs1 + nv_subs2 + lpw_subs + ['none', 'train']

    AllDS = selDataset(AllDS, ['OpenEDS', 'UnityEyes', 'NVGaze', 'LPW', 'riteyes_general'])
    AllDS = selSubset(AllDS, subsets)
    dataDiv_obj = generate_fileList(AllDS, mode='vanilla', notest=True)
    trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), 0.5)
    validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), 0.5)
    '''
    f = os.path.join('curObjects', 'cond_0.pkl')
    trainObj, validObj, _ = pickle.load(open(f, 'rb'))
    trainObj.path2data = path2h5

    trainLoader = DataLoader(trainObj,
                             batch_size=16,
                             shuffle=True,
                             num_workers=8,
                             drop_last=True)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    totTime = []
    startTime = time.time()
    for bt, data in enumerate(trainLoader):
        I, mask, spatialWeights, distMap, pupil_center, elPts, elNorm, cond, imInfo = data
        hMaps = points_to_heatmap(elPts, 2, I.shape[2:])
        dispI = generateImageGrid(I.squeeze().numpy(),
                                  mask.numpy(),
                                  hMaps.numpy(),
                                  elNorm.numpy(),
                                  pupil_center.numpy(),
                                  cond.numpy())
        
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
