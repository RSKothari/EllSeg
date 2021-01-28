#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:16:57 2019

@author: rakshit

This code extracts the following datasets:
else, excuse - Wolfgang Fuhl
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
import matplotlib.pyplot as plt

sys.path.append('..')
from helperfunctions import generateEmptyStorage, mypause

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Display labelled images', type=int, default=1)
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

print('Extracting Fuhl')

PATH_DIR = os.path.join(args.path2ds, 'Fuhl')
PATH_DS = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')
list_ds = ['data set I', 'data set II', 'data set III', 'data set IV',
           'data set IX', 'data set V', 'data set VI', 'data set VII',
           'data set VIII', 'data set X', 'data set XI', 'data set XII',
           'data set XIII', 'data set XIV', 'data set XIX', 'data set XVI',
           'data set XVII', 'data set XVIII', 'data set XX', 'data set XXI',
           'data set XXII', 'data set XXIII', 'data set XV', 'data set XXIV']

sc = (640.0/384.0)
Image_counter = 0.0
ds_num = 0

def fix_pupil_loc(p, res):
    # res: [H, W]
    p[0] = 0.5*p[0]
    p[1] = res[0] - 0.5*p[1]
    return p

for name in list_ds:
    # Ignore the first row and column.
    # Columns: [index, p_x, p_y]

    # Read pupil data from the published dataset
    PupilData = np.genfromtxt(os.path.join(PATH_DIR, name+'.txt'), delimiter=' ')[1:,1:]

    listFiles = glob.glob(os.path.join(PATH_DIR, name, '*.png'))
    imNames = list(map(os.path.basename, listFiles))
    imNames = list(map(os.path.splitext, imNames))
    imNames, _ = list(zip(*imNames))
    imNames = np.array(list(map(int, imNames)))

    # Generate an empty data container
    Data, keydict = generateEmptyStorage(name='Fuhl', subset=name)

    ds_name = keydict['dataset'] + '_' + keydict['subset'] + '_' + str(ds_num)

    if not noDisp:
        fig, plts = plt.subplots(1,1)
    for i in range(0, PupilData.shape[0]):
        iNum = PupilData[i, 0]

        # Assert unique image names
        loc = imNames == iNum
        assert sum(loc) == 1, "Error. Only one file should have that number"

        loc = np.where(loc)[0]
        path2im = listFiles[int(loc)]
        imStr = os.path.split(path2im)[1]

        # Read image and upscale
        I = cv2.imread(path2im, 0)
        I = cv2.resize(I, (640, 480), cv2.INTER_LANCZOS4)

        pupil_loc = copy.deepcopy(PupilData[i, 1:]*sc)
        pupil_loc = fix_pupil_loc(pupil_loc, I.shape) # Fix pupil position

        Data['Images'].append(I)
        Data['pupil_loc'].append(pupil_loc) # Fix in records too
        Data['Info'].append(imStr)

        keydict['Info'].append(imStr)
        keydict['resolution'].append(I.shape)
        keydict['archive'].append(ds_name)
        keydict['pupil_loc'].append(pupil_loc) # Fix in records too

        Image_counter = Image_counter + 1
        if not noDisp:
            if i == 0:
                cI = plts.imshow(I)
                cX = plts.scatter(pupil_loc[0], pupil_loc[1])
                plt.show()
                plt.pause(.01)
            else:
                newLoc = np.array([pupil_loc[0], pupil_loc[1]])
                cI.set_data(I)
                cX.set_offsets(newLoc)
                mypause(0.01)


    Data['Images'] = np.stack(Data['Images'], axis=0)
    Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
    keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
    keydict['archive'] = np.stack(keydict['archive'], axis=0)
    keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)

    # Save data
    dd.io.save(os.path.join(PATH_DS, str(ds_name)+'.h5'), Data)
    scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)
    ds_num=ds_num+1