#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:16:57 2019

@author: rakshit
"""
import os
import cv2
import glob
import json
import argparse
import matplotlib
import numpy as np
import deepdish as dd
import scipy.io as scio

from ast import literal_eval
from matplotlib.patches import Ellipse
from skimage.morphology.convex_hull import grid_points_in_poly
from RITEyes_helper.helperfunctions import ElliFit, mypause, my_ellipse, generateEmptyStorage

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int)
parser.add_argument('--path2ds', help='Path to dataset', type=str)
args = parser.parse_args()
if args.noDisp:
    print('No graphics')
else:
    args.noDisp = False
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

#ds_num = 161
ds_num = 0

args.path2ds = '/media/rakshit/tank/Dataset'
PATH_DIR = os.path.join(args.path2ds, 'UnityEyes', 'imgs')
PATH_DS = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')
list_ds = glob.glob(os.path.join(PATH_DIR, '*.jpg'))

N = 2000
opSize = (640, 480)
sc = 1/2 # Modify this to control how much to extract
x_sc = int(sc*opSize[0])
y_sc = int(sc*opSize[1])
x_crop = int(0.5*x_sc)
y_crop = int(0.5*y_sc)


for i in range(0, 5):
    ds_name = 'UnityEyes_'+str(i)+'_'+str(ds_num)
    Image_counter = 0.0

    if not args.noDisp:
        fig, plts = plt.subplots(1,1)

    Data, keydict = generateEmptyStorage(name='UnityEyes', subset='UnityEyes_{}'.format(i))

    for fName in list_ds[i*N:(i+1)*N]:
        path2im = os.path.split(fName)[0]
        imNum = os.path.splitext(os.path.split(fName)[1])[0]
        Info = json.load(open(os.path.join(path2im, imNum+'.json'), 'r'))

        # Read image
        I = cv2.imread(fName, 0)

        # Read sclera points
        pts_sclera = [np.asarray(literal_eval(ele)) for ele in Info['interior_margin_2d']]
        pts_sclera = np.stack(pts_sclera, axis=0)[:, :2]
        pts_sclera[:, 1] = 1080 - pts_sclera[:, 1]

        # Get sclera extremes and crop the image
        xc = int(np.mean([np.min(pts_sclera[:, 0]), np.max(pts_sclera[:, 0])]))
        yc = int(np.mean([np.min(pts_sclera[:, 1]), np.max(pts_sclera[:, 1])]))
        I_cropped = I[yc-y_crop:yc+y_crop, xc-x_crop:xc+x_crop]

        # Calculate shifts
        x_shift = x_crop - xc
        y_shift = y_crop - yc

        # Create scleral mask
        pts_sclera = pts_sclera + np.array([x_shift, y_shift])
        sclera_mask = grid_points_in_poly(I_cropped.shape, np.flip(pts_sclera, axis=1))

        # Read Iris points
        pts_iris = [np.asarray(literal_eval(ele)) for ele in Info['iris_2d']]
        pts_iris = np.stack(pts_iris, axis=0)[:, :2]
        pts_iris[:, 1] = 1080 - pts_iris[:, 1]

        # Shift Iris points
        pts_iris = pts_iris + np.array([x_shift, y_shift])

        # Normalize points, fit ellipses and find Iris mask
        irisFit = ElliFit(**{'data': pts_iris})
        iris_fit_error = my_ellipse(irisFit.model).verify(pts_iris)
        iris_mask = grid_points_in_poly(I_cropped.shape, np.flip(pts_iris, axis=1)).astype(np.uint8)

        # Create an iris mask
        mask = iris_mask.astype(np.uint8)*sclera_mask.astype(np.uint8) + sclera_mask.astype(np.uint8)

        # Upscale cropped images and fits
        I_cropped = cv2.resize(I_cropped, opSize, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, opSize, interpolation=cv2.INTER_NEAREST)
        iris_mask = cv2.resize(iris_mask, opSize, interpolation=cv2.INTER_NEAREST)
        irisFit.model[:-1] = [irisFit.model[i]*(1/sc) for i in range(0, 4)]
        
        # Shift the ellipse mask
        Data['Images'].append(I_cropped)
        Data['Masks'].append(mask)
        Data['Masks_noSkin'].append(iris_mask*2)
        Data['pupil_loc'].append(np.array([-1, -1]))
        Data['Fits']['iris'].append(irisFit.model)
        Data['Fits']['pupil'].append(np.array([-1, -1, -1, -1, -1]))

        keydict['pupil_loc'].append(np.array([-1, -1]))
        keydict['resolution'].append(I_cropped.shape)
        keydict['archive'].append(ds_name)

        if not args.noDisp:
            if Image_counter == 0:
                cI = plts.imshow(I_cropped, cmap='gray')
                cE = Ellipse(tuple(irisFit.model[:2]),
                                   2*irisFit.model[2],
                                   2*irisFit.model[3],
                                   angle=np.rad2deg(irisFit.model[-1]))
                cE.set_facecolor('None')
                cE.set_edgecolor((0.0, 1.0, 0.0))
                cM = plts.imshow(mask, alpha=0.3)
                plts.add_patch(cE)
                plt.show()
                plt.pause(.01)
            else:
                cE.center = tuple(irisFit.model[:2])
                cE.angle = np.rad2deg(irisFit.model[-1])
                cE.width = 2*irisFit.model[2]
                cE.height = 2*irisFit.model[3]
                cI.set_data(I_cropped)
                cM.set_data(mask)
                mypause(0.01)
        Image_counter = Image_counter + 1

    Data['Images'] = np.stack(Data['Images'], axis=0)
    Data['Masks'] = np.stack(Data['Masks'], axis=0)
    Data['Masks_noSkin'] = np.stack(Data['Masks_noSkin'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    
    keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
    keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
    keydict['archive'] = np.stack(keydict['archive'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    Data['Fits']['iris'] = np.stack(Data['Fits']['iris'], axis=0)
    Data['Fits']['pupil'] = np.stack(Data['Fits']['pupil'], axis=0)

    # Save data
    dd.io.save(os.path.join(PATH_DS, ds_name+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, ds_name+'.mat'), keydict, appendmat=True)
    ds_num=ds_num+1