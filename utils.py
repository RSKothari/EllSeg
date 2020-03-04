#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:17:32 2020

@author: rakshit
"""

# This file contains definitions which are not available for regular scenarios.
# For general purposes functions, classes and operations - please use RITEyes.
import copy
import torch
import numpy as np

from torchvision.utils import make_grid
from skimage.draw import circle
from sklearn import metrics

def getSeg_metrics(y_true, y_pred, cond):
    '''
    Iterate over each batch and identify which classes are present. If no
    class is present, i.e. all 0, then ignore that score from the average.
    '''
    cond = cond.astype(np.bool)
    B = y_true.shape[0]
    score_list = []
    for i in range(0, B):
        labels_present = np.unique(y_true[i, ...])
        score_vals = np.empty((3, ))
        score_vals[:] = np.nan
        if not cond[i, 1]:
            score = metrics.jaccard_score(y_true[i, ...].reshape(-1),
                                          y_pred[i, ...].reshape(-1),
                                          labels=labels_present,
                                          average=None)
            # Assign score to relevant location
            for j, val in np.ndenumerate(labels_present):
                score_vals[val] = score[j]
        score_list.append(score_vals)
    score_list = np.stack(score_list, axis=0)
    score_list = score_list[~cond[:, 1], :] # Only select valid entries
    meanIOU = np.mean(np.mean(score_list, axis=1))
    perClassIOU = np.mean(score_list, axis=0)
    return meanIOU, perClassIOU

def getPoint_metric(y_true, y_pred, cond):
    cond = cond.astype(np.bool)
    flag = (~cond[:, 0]).astype(np.float)
    dist = metrics.pairwise_distances(y_true, y_pred, metric='euclidean')
    dist = flag*np.diag(dist)
    return np.sum(dist)/np.sum(flag)

def generateImageGrid(I, mask, pupil_center, cond):
    I_o = []
    for i in range(0, cond.shape[0]):
        im = I[i, ...].squeeze()
        im = np.stack([im for i in range(0, 3)], axis=2)
        
        if not cond[i, 1]:
            # If masks exists
            rr, cc = np.where(mask[i, ...] == 1)
            im[rr, cc, ...] = np.array([1, -1, -1]) # Red
            rr, cc = np.where(mask[i, ...] == 2)
            im[rr, cc, ...] = np.array([1, -1, 1])
        
        if not cond[i, 0]:
            # If pupil center exists
            rr, cc = circle(pupil_center[i, 1].clip(5, im.shape[1]-5),
                            pupil_center[i, 0].clip(5, im.shape[0]-5),
                            5)
            im[rr, cc, ...] = 1
        I_o.append(im)
    I_o = np.stack(I_o, axis=0)
    I_o = np.moveaxis(I_o, 3, 1)
    I_o = make_grid(torch.from_numpy(I_o), nrow=4)
    I_o = I_o - I_o.min()
    I_o = I_o/I_o.max()
    return I_o

def normPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    pts_o[:, 0] = 2*(pts_o[:, 0]/sz[1]) - 1
    pts_o[:, 1] = 2*(pts_o[:, 1]/sz[0]) - 1
    return pts_o

def unnormPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    pts_o[:, 0] = 0.5*sz[1]*(pts_o[:, 0] + 1)
    pts_o[:, 1] = 0.5*sz[0]*(pts_o[:, 1] + 1)
    return pts_o