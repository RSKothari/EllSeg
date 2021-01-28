#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:16:57 2019

@author: rakshit
"""
import os
import cv2
import sys
import glob
import copy
import argparse
import matplotlib
import numpy as np
import deepdish as dd
import scipy.io as scio

sys.path.append('..')

from PIL import Image
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse

from helperfunctions import generateEmptyStorage, getValidPoints
from helperfunctions import ransac, ElliFit, my_ellipse

import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds',
                    help='Path to dataset',
                    type=str,
                    default='/media/rakshit/Monster/Datasets')
args = parser.parse_args()

if args.noDisp:
    noDisp = True
    print('No graphics')
else:
    noDisp = False
    print('Showing figures')

print('Extracting RITEyes: s-general')

gui_env = ['Qt5Agg','WXAgg','TKAgg','GTKAgg']
for gui in gui_env:
    try:
        print("testing: {}".format(gui))
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue

print("Using: {}".format(matplotlib.get_backend()))
plt.ion()

PATH_DIR = os.path.join(args.path2ds, 's-general')
PATH_DS = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')

Image_counter = 0.0
ds_num = 0

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def quantizeMask(wSkin_mask, I):
    # Quantize for pupil and iris.
    # Pupil is red
    # Iris is green
    # Scelra is blue
    r, c, _ = I.shape
    x, y = np.meshgrid(np.arange(0, c), np.arange(0, r))
    mask = np.zeros((r, c))
    mask_red = np.bitwise_and(
            I[:,:,0]>=248, I[:,:,1]==0, I[:,:,2]==0)
    mask_green = np.bitwise_and(
            I[:,:,0]==0, I[:,:,1]>=248, I[:,:,2]==0)
    N_pupil = np.sum(mask_red)
    N_iris = np.sum(mask_green)
    noPupil = False if N_pupil > 20 else True
    noIris = False if N_iris > 20 else True
    # Pupil and Iris regions, absolutely no sclera
    if not noPupil and not noIris:
        initarr = np.array([[0,0,0],
                            [0,0,255],
                            [0,255,0],
                            [255,0,0]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=4,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]<128) & (wSkin_mask[:,:,1]<128) & (wSkin_mask[:,:,2]<128)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0

    if noPupil and not noIris:
        initarr = np.array([[0,0,0],
                            [0,0,255],
                            [0,255,0]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=3,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]<128) & (wSkin_mask[:,:,1]<128) & (wSkin_mask[:,:,2]<128)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0

    if not noPupil and noIris:
        initarr = np.array([[0,0,0],
                            [0,0,255],
                            [255,0,0]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=3,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask[mask == 2] = 3 # Should actually be 3 for pupil locations
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]<128) & (wSkin_mask[:,:,1]<128) & (wSkin_mask[:,:,2]<128)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0

    if noPupil and noIris:
        initarr = np.array([[0,0,0],
                            [0,0,255]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=2,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]<128) & (wSkin_mask[:,:,1]<128) & (wSkin_mask[:,:,2]<128)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0
    return (wSkin_mask, mask)

list_ds = [ele for ele in os.listdir(PATH_DIR) if os.path.isdir(os.path.join(PATH_DIR, ele))]
list_ds.remove('3d')

for fName in list_ds:
    warnings.filterwarnings("error")
    PATH_IMAGES = os.path.join(PATH_DIR, fName, 'synthetic')
    PATH_MASK_SKIN = os.path.join(PATH_DIR, fName, 'mask-withskin')
    PATH_MASK_NOSKIN = os.path.join(PATH_DIR, fName, 'mask-withoutskin-noglasses')

    imList = glob.glob(os.path.join(PATH_IMAGES, '*.tif'))

    if not noDisp:
        fig, plts = plt.subplots(1,1)

    Data, keydict = generateEmptyStorage(name='riteyes_general', subset='riteyes_general_'+fName)
    ds_name = 'riteyes_general_{}_{}'.format(fName, ds_num)
    fr_num = 0

    for ele in imList:
        imName_withext = os.path.split(ele)[1]
        imName = os.path.splitext(ele)[0]

        path2im = os.path.join(PATH_IMAGES, imName_withext)
        path2mask = os.path.join(PATH_MASK_SKIN, imName_withext)
        path2mask_woskin = os.path.join(PATH_MASK_NOSKIN, imName_withext)
        try:
            I = np.asarray(Image.open(path2im).convert('L'))
            maskIm = np.asarray(Image.open(path2mask).convert('RGB'))
            maskIm_woskin = np.asarray(Image.open(path2mask_woskin).convert('RGB'))

            I = cv2.resize(I, (640, 480), interpolation=cv2.INTER_CUBIC)
            maskIm = cv2.resize(maskIm, (640, 480), interpolation=cv2.INTER_NEAREST)
            maskIm_woskin = cv2.resize(maskIm_woskin, (640, 480), interpolation=cv2.INTER_NEAREST)

            maskIm, maskIm_woskin = quantizeMask(maskIm, maskIm_woskin)
        except:
            print('Corrupt data found in {}.'.format(ele))
            continue

        pupilPts, irisPts = getValidPoints(maskIm_woskin)

        # Pupil ellipse fit
        model_pupil = ransac(pupilPts, ElliFit, 15, 40, 5e-3, 15).loop()
        pupil_fit_error = my_ellipse(model_pupil.model).verify(pupilPts)

        r, c = np.where(maskIm_woskin == 2)
        pupil_loc = model_pupil.model[:2] if pupil_fit_error < 0.05 else np.stack([np.mean(c), np.mean(r)], axis=0)

        # Iris ellipse fit
        model_iris = ransac(irisPts, ElliFit, 15, 40, 5e-3, 15).loop()
        iris_fit_error = my_ellipse(model_iris.model).verify(irisPts)

        if (pupil_fit_error > 0.05) | (iris_fit_error > 0.05):
            print('Skipping: {}'.format(imName))
            continue

        Data['Images'].append(I)
        Data['Masks'].append(maskIm)
        Data['Masks_noSkin'].append(maskIm_woskin)
        Data['Info'].append(imName)
        Data['pupil_loc'].append(pupil_loc)
        keydict['resolution'].append(I.shape)
        keydict['archive'].append(ds_name)
        keydict['pupil_loc'].append(pupil_loc)

        Data['Fits']['pupil'].append(model_pupil.model)
        Data['Fits']['iris'].append(model_iris.model)

        fr_num += 1
        if not noDisp:
            if fr_num == 1:
                cE = Ellipse(tuple(pupil_loc),
                             2*model_pupil.model[2],
                             2*model_pupil.model[3],
                             angle=np.rad2deg(model_pupil.model[-1]))
                cL = Ellipse(tuple(model_iris.model[0:2]),
                                   2*model_iris.model[2],
                                   2*model_iris.model[3],
                                   np.rad2deg(model_iris.model[4]))
                cE.set_facecolor('None')
                cE.set_edgecolor((1.0, 0.0, 0.0))
                cL.set_facecolor('None')
                cL.set_edgecolor((0.0, 1.0, 0.0))
                cI = plts.imshow(I)
                cM = plts.imshow(maskIm, alpha=0.5)
                cX = plts.scatter(pupil_loc[0], pupil_loc[1])
                plts.add_patch(cE)
                plts.add_patch(cL)
                plt.show()
                plt.pause(.01)
            else:
                cE.center = tuple(pupil_loc)
                cE.angle = np.rad2deg(model_pupil.model[-1])
                cE.width = 2*model_pupil.model[2]
                cE.height = 2*model_pupil.model[3]
                cL.center = tuple(model_iris.model[0:2])
                cL.width = 2*model_iris.model[2]
                cL.height = 2*model_iris.model[3]
                cL.angle = np.rad2deg(model_iris.model[-1])
                newLoc = np.array([pupil_loc[0], pupil_loc[1]])
                cI.set_data(I)
                cM.set_data(maskIm)
                cX.set_offsets(newLoc)
                mypause(0.01)

    keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
    keydict['archive'] = np.stack(keydict['archive'], axis=0)
    keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    Data['Images'] = np.stack(Data['Images'], axis=0)
    Data['Masks'] = np.stack(Data['Masks'], axis=0)
    Data['Masks_noSkin'] = np.stack(Data['Masks_noSkin'], axis=0)
    Data['Fits']['pupil'] = np.stack(Data['Fits']['pupil'], axis=0)
    Data['Fits']['iris'] = np.stack(Data['Fits']['iris'], axis=0)

    warnings.filterwarnings("ignore")

    # Save data
    dd.io.save(os.path.join(PATH_DS, str(ds_name)+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)
    ds_num=ds_num+1