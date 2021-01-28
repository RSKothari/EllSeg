#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:21:36 2019

@author: rakshit
"""
# Confirmed code works perfectly. Do not display.
import os
import cv2
import sys
import json
import argparse
import matplotlib
import numpy as np
import deepdish as dd
import scipy.io as scio
from matplotlib.patches import Ellipse
from skimage.draw import ellipse as drawEllipse

sys.path.append('..')

from helperfunctions import ransac, ElliFit, my_ellipse
from helperfunctions import generateEmptyStorage, getValidPoints

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

ds_num = 0
PATH_OPENEDS = os.path.join(args.path2ds, 'OpenEDS')
PATH_DIR = os.path.join(args.path2ds, 'OpenEDS', 'Semantic_Segmentation_Dataset')
PATH_DS = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')

print('Extracting OpenEDS')

# Don't append the test set.
listDir = os.listdir(PATH_DIR)
listDir.remove('test')
for dirCond in listDir:
    ds_name = 'OpenEDS_{}_{}'.format(dirCond, ds_num)

    print('Opening the {} folder'.format(dirCond))

    # Read UID
    path2json = os.path.join(PATH_OPENEDS, 'OpenEDS_{}_userID_mapping_to_images.json'.format(dirCond))
    im2UID = json.load(open(path2json, 'r'))

    PATH_IMAGES = os.path.join(PATH_DIR, dirCond, 'images')
    PATH_LABELS = os.path.join(PATH_DIR, dirCond, 'labels')
    PATH_FITS = os.path.join(PATH_DIR, dirCond, 'fits')
    listIm = os.listdir(PATH_IMAGES)

    Data, keydict = generateEmptyStorage(name='OpenEDS', subset=dirCond)

    i = 0
    if not noDisp:
        fig, plts = plt.subplots(1,1)

    for pData in im2UID:
        # Image number and UID for each person
        listIm = pData['semantic_segmenation_images']
        pid = int(pData['id'].replace('U', '')) - 111
        for imName_full in listIm:
            imName, _ = os.path.splitext(imName_full)

            # Do not save images without a proper ellipse and iris fit
            # Load image, label map and fits
            I = cv2.imread(os.path.join(PATH_IMAGES, imName_full), 0)
            LabelMat = np.load(os.path.join(PATH_LABELS, imName+'.npy'))

            #%% Make sure images are 640x480
            r = np.where(LabelMat)[0]
            c = int(0.5*(np.max(r) + np.min(r)))
            top, bot = (0, c+150-(c-150)) if c-150<0 else (c-150, c+150)

            I = I[top:bot, :]
            LabelMat = LabelMat[top:bot, :]
            I = cv2.resize(I, (640, 480), interpolation=cv2.INTER_LANCZOS4)
            LabelMat = cv2.resize(LabelMat, (640, 480), interpolation=cv2.INTER_NEAREST)
            #%%

            pupilPts, irisPts = getValidPoints(LabelMat)
            if np.sum(LabelMat == 3) > 150 and type(pupilPts) is not list:
                model_pupil = ransac(pupilPts, ElliFit, 15, 40, 5e-3, 15).loop()
                pupil_fit_error = my_ellipse(model_pupil.model).verify(pupilPts)
            else:
                print('Not enough pupil points')
                model_pupil = type('model', (object, ), {})
                model_pupil.model = np.array([-1, -1, -1, -1, -1])
                pupil_fit_error = np.inf

            if np.sum(LabelMat == 2) > 200 and type(irisPts) is not list:
                model_iris = ransac(irisPts, ElliFit, 15, 40, 5e-3, 15).loop()
                iris_fit_error = my_ellipse(model_iris.model).verify(irisPts)
            else:
                print('Not enough iris points')
                model_iris = type('model', (object, ), {})
                model_iris.model = np.array([-1, -1, -1, -1, -1])
                model_iris.Phi = np.array([-1, -1, -1, -1, -1])
                iris_fit_error = np.inf

            if pupil_fit_error >= 0.1:
                print('Not recording pupil. Unacceptable fit.')
                print('Pupil fit error: {}'.format(pupil_fit_error))
                model_pupil.model = np.array([-1, -1, -1, -1, -1])

            if iris_fit_error >= 0.1:
                print('Not recording iris. Unacceptable fit.')
                print('Iris fit error: {}'.format(iris_fit_error))
                model_iris.model = np.array([-1, -1, -1, -1, -1])

            pupil_loc = model_pupil.model[:2]

            # Draw mask no skin
            rr, cc = drawEllipse(pupil_loc[1],
                                 pupil_loc[0],
                                 model_pupil.model[3],
                                 model_pupil.model[2],
                                 rotation=-model_pupil.model[-1])
            pupMask = np.zeros_like(I)
            pupMask[rr.clip(0, I.shape[0]-1), cc.clip(0, I.shape[1]-1)] = 1
            rr, cc = drawEllipse(model_iris.model[1],
                                  model_iris.model[0],
                                  model_iris.model[3],
                                  model_iris.model[2],
                                  rotation=-model_iris.model[-1])
            iriMask = np.zeros_like(I)
            iriMask[rr.clip(0, I.shape[0]-1), cc.clip(0, I.shape[1]-1)] = 1

            if (np.any(pupMask) and np.any(iriMask)) and ((pupil_fit_error<0.1) and (iris_fit_error<0.1)):
                mask_woSkin = 2*iriMask + pupMask # Iris = 2, Pupil = 3
            else:
                # Neither fit exists, mask should be -1s.
                print('Found bad mask: {}'.format(imName))
                mask_woSkin = -np.ones(I.shape)
                continue

            # Add model information
            keydict['archive'].append(ds_name)
            keydict['resolution'].append(I.shape)
            keydict['pupil_loc'].append(pupil_loc)

            # Append images and label map
            Data['Images'].append(I)
            Data['Info'].append(imName_full) # Train or valid
            Data['Masks'].append(LabelMat)
            Data['Masks_noSkin'].append(mask_woSkin)
            Data['pupil_loc'].append(pupil_loc)

            # Append fits
            Data['Fits']['pupil'].append(model_pupil.model)
            Data['Fits']['iris'].append(model_iris.model)

            keydict['Fits']['pupil'].append(model_pupil.model)
            keydict['Fits']['iris'].append(model_iris.model)

            if not noDisp:
                if i == 0:
                    cE = Ellipse(tuple(pupil_loc),
                                 2*model_pupil.model[2],
                                 2*model_pupil.model[3],
                                 angle=np.rad2deg(model_pupil.model[4]))
                    cL = Ellipse(tuple(model_iris.model[0:2]),
                                       2*model_iris.model[2],
                                       2*model_iris.model[3],
                                       np.rad2deg(model_iris.model[4]))
                    cE.set_facecolor('None')
                    cE.set_edgecolor((1.0, 0.0, 0.0))
                    cL.set_facecolor('None')
                    cL.set_edgecolor((0.0, 1.0, 0.0))
                    cI = plts.imshow(I)
                    cM = plts.imshow(mask_woSkin, alpha=0.5)
                    plts.add_patch(cE)
                    plts.add_patch(cL)
                    plt.show()
                    plt.pause(.01)
                else:
                    cE.center = tuple(pupil_loc)
                    cE.angle = np.rad2deg(model_pupil.model[4])
                    cE.width = 2*model_pupil.model[2]
                    cE.height = 2*model_pupil.model[3]
                    cL.center = tuple(model_iris.model[0:2])
                    cL.width = 2*model_iris.model[2]
                    cL.height = 2*model_iris.model[3]
                    cL.angle = np.rad2deg(model_iris.model[-1])
                    cI.set_data(I)
                    cM.set_data(mask_woSkin)
                    mypause(0.01)
            i = i + 1
    print('{} images: {}'.format(dirCond, i))

    # Stack data
    Data['Images'] = np.stack(Data['Images'], axis=0)
    Data['Masks'] = np.stack(Data['Masks'], axis=0)
    Data['Masks_noSkin'] = np.stack(Data['Masks_noSkin'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    Data['Fits']['pupil'] = np.stack(Data['Fits']['pupil'], axis=0)
    Data['Fits']['iris'] = np.stack(Data['Fits']['iris'], axis=0)

    keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
    keydict['archive'] = np.stack(keydict['archive'], axis=0)
    keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)

    # Save data
    dd.io.save(os.path.join(PATH_DS, ds_name+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)
    ds_num=ds_num+1