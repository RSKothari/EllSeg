#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:16:57 2019

@author: rakshit
"""
import os
import re
import io
import cv2
import sys
import copy
import zipfile
import argparse
import matplotlib
import numpy as np
import deepdish as dd
import scipy.io as scio

from PIL import Image
from random import shuffle
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse

import warnings
warnings.filterwarnings("error")

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds',
                    help='Path to dataset',
                    type=str,
                    default='/media/rakshit/Monster/Datasets')
args = parser.parse_args()

#sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.abspath('..')))
from helperfunctions import ransac, ElliFit, my_ellipse
from helperfunctions import generateEmptyStorage

if args.noDisp:
    noDisp = True
    print('No graphics')
else:
    noDisp = False
    print('Showing figures')

print('Extracting NVGaze')

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
PATH_DIR = os.path.join(args.path2ds, 'NVGaze', 'synthetic_dataset')
PATH_DS = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')
list_ds = [ele for ele in os.listdir(PATH_DIR) if os.path.isdir(os.path.join(PATH_DIR, ele))]

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

def readFormattedText(path2file, ignoreLines):
    data = []
    count = 0
    f = open(path2file, 'r')
    for line in f:
        if count > ignoreLines:
            d = [d for d in line.split(',')]
            data.append(d)
        count = count + 1
    f.close()
    return data

def preProcessNV(I, datatype, sc):
    xs, ys = I.shape[:2]
    I = cv2.resize(I, (np.uint(ys*sc), np.uint(xs*sc)),
                   interpolation=cv2.INTER_LANCZOS4).astype(datatype)
    return I


def quantizeMask(wSkin_mask, I):
    # Quantize for pupil and iris.
    # Pupil is green
    # Iris is yellow
    # Scelra is amber
    r, c, _ = I.shape
    x, y = np.meshgrid(np.arange(0, c), np.arange(0, r))
    mask = np.zeros((r, c))
    mask_yellow = np.bitwise_and(
            I[:,:,0]==255, I[:,:,1]==255, I[:,:,2]==0)
    mask_green = np.bitwise_and(
            I[:,:,0]==0, I[:,:,1]==255, I[:,:,2]==0)
    N_pupil = np.sum(mask_green)
    N_iris = np.sum(mask_yellow)

    noPupil = False if N_pupil > 20 else True
    noIris = False if N_iris > 20 else True

    # Pupil and Iris regions, absolutely no sclera
    if not noPupil and not noIris:
        initarr = np.array([[0,0,0],
                            [255,215,0],
                            [255,255,0],
                            [0,255,0]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=4,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]==255) & (wSkin_mask[:,:,1]==0) & (wSkin_mask[:,:,2]==0)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0

    if noPupil and not noIris:
        initarr = np.array([[0,0,0],
                            [255,215,0],
                            [255,255,0]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=3,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]==255) & (wSkin_mask[:,:,1]==0) & (wSkin_mask[:,:,2]==0)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0

    if not noPupil and noIris:
        initarr = np.array([[0,0,0],
                            [255,215,0],
                            [0,255,0]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=3,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask[mask == 2] = 3 # Should actually be 3 for pupil locations
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]==255) & (wSkin_mask[:,:,1]==0) & (wSkin_mask[:,:,2]==0)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0

    if noPupil and noIris:
        initarr = np.array([[0,0,0],
                            [255,215,0]])
        feats = I.reshape(-1, 3)
        KM = KMeans(n_clusters=2,
                    max_iter=1000,
                    tol=1e-6, n_init=1,
                    init=initarr).fit(feats)
        mask = KM.predict(feats)
        mask = mask.reshape(r, c)
        loc = (wSkin_mask[:,:,0]==255) & (wSkin_mask[:,:,1]==0) & (wSkin_mask[:,:,2]==0)
        wSkin_mask = copy.deepcopy(mask)
        wSkin_mask[loc] = 0
    return (wSkin_mask, mask)

for fName in list_ds:
    warnings.filterwarnings("error")
    ds_name = 'NVGaze'+'_'+fName+'_'+str(ds_num)

    # Ignore the first row and column.
    # Columns: [index, p_x, p_y]
    ZipObj = zipfile.ZipFile(os.path.join(PATH_DIR, fName, 'footage_image_data.zip'))
    imList = [ele for ele in ZipObj.namelist() if 'type_img_frame' in ele]
    shuffle(imList)

    if not noDisp:
        fig, plts = plt.subplots(1,1)

    Data, keydict = generateEmptyStorage(name='NVGaze', subset=fName)

    fr_num = 0

    for boo in imList[:500]:
        # Read pupil  info
        imNum_str = re.findall('\d+', boo)[0]
        str_imName = 'type_img_frame_{}.png'.format(imNum_str)
        str_imName_Mask = 'type_maskWithSkin_frame_{}.png'.format(imNum_str)
        str_imName_Mask_woSkin = 'type_maskWithoutSkin_frame_{}.png'.format(imNum_str)

        I = ZipObj.read(str_imName)
        I = np.array(Image.open(io.BytesIO(I))).astype(np.double)
        I = preProcessNV(I, np.uint8, 0.5)

        mask = ZipObj.read(str_imName_Mask)
        mask = np.array(Image.open(io.BytesIO(mask)))
        mask = preProcessNV(mask, np.uint8, 0.5)

        fullmask = ZipObj.read(str_imName_Mask_woSkin)
        fullmask = np.array(Image.open(io.BytesIO(fullmask)))
        fullmask = preProcessNV(fullmask, np.uint8, 0.5)
        mask, mask_noSkin = quantizeMask(mask, fullmask)

        if not np.any(mask_noSkin):
            print('Error in mask. Package: {}. Idx: {}'.format(fName, boo))
            mask_noSkin = -np.ones_like(I)

        Data['Images'].append(I)
        Data['Masks'].append(mask)
        Data['Masks_noSkin'].append(mask_noSkin)
        Data['Info'].append(str_imName)
        keydict['resolution'].append(I.shape)
        keydict['archive'].append(ds_name)

        temp = 255*(mask_noSkin == 3)
        edge = cv2.Canny(temp.astype(np.uint8), 10, 150) + cv2.Canny((255-temp).astype(np.uint8), 10, 150)
        r, c = np.where(edge)
        temp_pts = np.stack([c, r], axis=1) # Improve readability
        model_pupil = ransac(temp_pts, ElliFit, 15, 10, 0.05, np.round(temp_pts.shape[0]/2)).loop()
        pupil_fit_error = my_ellipse(model_pupil.model).verify(temp_pts)

        pupil_loc = model_pupil.model[:2]

        temp = 255*((mask_noSkin == 2) | (mask_noSkin == 3))
        edge = cv2.Canny(temp.astype(np.uint8), 10, 150)+ cv2.Canny((255-temp).astype(np.uint8), 10, 150)
        r, c = np.where(edge)
        temp_pts = np.stack([c, r], axis=1)
        model_iris = ransac(temp_pts, ElliFit, 15, 10, 0.05, np.round(temp_pts.shape[0]/2)).loop()
        iris_fit_error = my_ellipse(model_iris.model).verify(temp_pts)

        Data['Fits']['pupil'].append(model_pupil.model)
        Data['Fits']['iris'].append(model_iris.model)

        keydict['pupil_loc'].append(pupil_loc)
        Data['pupil_loc'].append(pupil_loc)

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
                                   np.rad2deg(model_iris.model[-1]))
                cE.set_facecolor('None')
                cE.set_edgecolor((1.0, 0.0, 0.0))
                cL.set_facecolor('None')
                cL.set_edgecolor((0.0, 1.0, 0.0))
                cI = plts.imshow(I)
                cM = plts.imshow(mask, alpha=0.5)
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
                cM.set_data(mask)
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

    # Save data
    warnings.filterwarnings("ignore")
    dd.io.save(os.path.join(PATH_DS, ds_name+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, ds_name+'.mat'), keydict, appendmat=True)
    ds_num=ds_num+1