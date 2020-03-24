#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:03:01 2019

@author: rakshit
"""

import os
import glob
import numpy as np
import deepdish as dd
import scipy.io as scio
import matplotlib.pyplot as plt
from skimage.draw import ellipse

path2labels = '/media/rakshit/tank/Dataset/OpenEDS/Semantic_Segmentation_Dataset/train/labels'
path2fits = '/media/rakshit/tank/Dataset/OpenEDS/Semantic_Segmentation_Dataset/train/fits'
path2h5 = '/media/rakshit/tank/Dataset/All/119.h5'

D = glob.glob(os.path.join(path2labels, '*.npy'))

fits = dd.io.load(path2h5, '/Fits')
ImName_h5 = dd.io.load(path2h5, '/Info')
masks = dd.io.load(path2h5, '/Masks')

lower_dist = []
upper_dist = []
xDev = []
yDev = []
corners_available = []
ratio_closure = []
ellipse_ang = []

for fid in D:
    fName = os.path.split(fid)[1]
    fName = os.path.splitext(fName)[0]

    #loc = np.where([True if fName in ele else False for ele in ImName_h5])[0]

    L = np.load(fid)
    '''
    iris_fit = fits['iris'][loc, :].squeeze() if type(fits['iris']) is not list else np.zeros((1, ))
    pupil_fit = fits['pupil'][loc, :].squeeze() if type(fits['pupil']) is not list else np.zeros((1, ))
    '''
    fit = scio.loadmat(os.path.join(path2fits, fName+'.mat'))

    iris_fit = fit['model_params_iris'].squeeze() if fit['model_params_iris'].size is not 0 else np.zeros((1, ))
    pupil_fit = fit['model_params_pupil'].squeeze() if fit['model_params_pupil'].size is not 0 else np.zeros((1, ))

    iris_loc = iris_fit[:2] if iris_fit.size > 1 else np.nan
    pupil_loc = pupil_fit[:2] if pupil_fit.size >1 else np.nan

    if (iris_loc is not np.nan) and (pupil_loc is not np.nan):
        I = np.zeros((640, 400))
        rr, cc = ellipse(iris_loc[1], iris_loc[0], iris_fit[2], iris_fit[3], (640, 400), iris_fit[4])
        I[rr, cc] = 1
        if np.sum(I) < 50:
            continue
        r_lid, c_lid = np.where(L != 0)
        r_iris, c_iris = np.where(I)

        dist1 = np.max(r_iris) - np.max(r_lid)
        dist2 = np.min(r_iris) - np.min(r_lid)

        ellipse_ang.append(np.rad2deg(pupil_fit[-1]))
        ratio_closure.append((np.max(r_lid) - np.min(r_lid))/(np.max(r_iris) - np.min(r_iris)))
        upper_dist.append(dist2)
        lower_dist.append(dist1)
        xDev.append(pupil_loc[0] - iris_loc[0])
        yDev.append(pupil_loc[1] - iris_loc[1])

#%% deviation vs ellipse angle

fig, plts = plt.subplots(1, 2, sharex=True, sharey=True)
plts[0].scatter(xDev, ellipse_ang)
plts[1].scatter(yDev, ellipse_ang)
plts[0].set_xlabel('Deviation')
plts[0].set_ylabel('Ellipse angle')
plts[0].set_title('Horizontal')
plts[1].set_xlabel('Deviation')
plts[1].set_ylabel('Ellipse angle')
plts[1].set_title('Vertical')

#%% ellipse angle vs upper and lower

fig, plts = plt.subplots(1, 2, sharex=True, sharey=False)
plts[0].scatter(ellipse_ang, upper_dist)
plts[1].scatter(ellipse_ang, lower_dist)
#plts[2].scatter(ellipse_ang, ratio_closure)
plts[0].set_ylabel('Distance (px)')
plts[0].set_xlabel('Ellipse angle (deg)')
plts[0].set_title('Upper')
plts[1].set_ylabel('Distance (px)')
plts[1].set_xlabel('Ellipse angle (deg)')
plts[1].set_title('Lower')
#plts[2].set_ylabel('closure ratio')
#plts[2].set_xlabel('Ellipse angle (deg)')
#plts[2].set_title('Dist eyelid / dist Iris')
