#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 07:29:05 2020

@author: rakshit
"""

import os
import cv2
import sys
import h5py
import pickle
import numpy as np

sys.path.append('..')

path2curObjs = '/home/rakshit/Documents/Python_Scripts/GIW_e2e/curObjects/baseline'
path2ds = '/media/rakshit/tank/Dataset/All'

ds_list = ['Fuhl', 'PupilNet', 'LPW', 'NVGaze', 'OpenEDS', 'riteyes_general']
cond = ['Natural', 'Constrained', 'Natural', 'Constrained', 'Natural', 'Constrained']

curObjs_list = os.listdir(path2curObjs)

def readImages(obj):
    archNums = np.unique(obj.imList[:, 1])
    I_list = []
    for archNum in np.nditer(archNums):
        f = h5py.File(os.path.join(path2ds, obj.arch[archNum]+'.h5'), 'r')
        im_ids = obj.imList[obj.imList[:, 1] == archNum, 0]
        loc = np.in1d(np.arange(f['Images'].shape[0]), im_ids)
        I = np.array(f['Images'][loc, ...]).astype(np.float32)
        I_list.append(normalize(I))
        f.close()
    return np.concatenate(I_list, axis=0)

def normalize(imgs):
    # Given a large amount of images, normalize and return
    L, H, W = imgs.shape
    mu = np.mean(imgs.reshape(L, -1), axis=1) # mu [L, ]
    std = np.std(imgs.reshape(L, -1), axis=1) # std [L, ]
    norm_data = imgs.reshape(L, -1) - np.stack([mu for i in range(H*W)], axis=1)
    norm_data = norm_data/np.stack([std for i in range(H*W)], axis=1)
    return norm_data

pxStats = {'name': [],
           'train': [],
           'valid': [],
           'test': []}

for ds_name in ds_list:
    print('Starting: {}'.format(ds_name))
    pxStats['name'].append(ds_name)
    path2curObj = os.path.join(path2curObjs, 'cond_'+ds_name+'.pkl')
    trainObj, validObj, testObj = pickle.load(open(path2curObj, 'rb'))
    
    # Extract train stats
    norm_data = readImages(trainObj)
    vals = np.apply_along_axis(lambda x: np.histogram(x,
                                                      range=(-4, 4),
                                                      bins=40),
                               axis=1,
                               arr=norm_data)
    pxStats['train'].append(vals)
    print('Train done: {}'.format(ds_name))
    
    # Extract valid stats
    norm_data = readImages(validObj)
    vals = np.apply_along_axis(lambda x: np.histogram(x,
                                                      range=(-4, 4),
                                                      bins=40),
                               axis=1,
                               arr=norm_data)
    pxStats['valid'].append(vals)
    print('Valid done: {}'.format(ds_name))
    
    # Extract test stats
    norm_data = readImages(testObj)
    vals = np.apply_along_axis(lambda x: np.histogram(x,
                                                      range=(-4, 4),
                                                      bins=40),
                               axis=1,
                               arr=norm_data)
    pxStats['test'].append(vals)
    print('Test done: {}'.format(ds_name))
    
    # Save out data
    f = open('statData.pkl','wb')
    pickle.dump(pxStats, f)
    f.close()
    print('Done: {}'.format(ds_name))