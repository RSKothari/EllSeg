#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:14:47 2020

@author: rakshit
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

idx = 102 # Other interesting figures

Data = h5py.File('/media/rakshit/tank/Dataset/All/OpenEDS_validation_1.h5', 'r')

I = Data['Images'][idx, ...]
mask_epSeg = Data['Masks'][idx, ...]
mask_elSeg = Data['Masks_noSkin'][idx, ...]
pupil_el = Data['Fits']['pupil'][idx, ...]
iris_el = Data['Fits']['iris'][idx, ...]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(I, cmap='gray', alpha=1)
axs[0].imshow(mask_epSeg, alpha=0.5)

axs[1].imshow(I, cmap='gray', alpha=1)
axs[1].imshow(mask_elSeg, alpha=0.5)

fig, axs = plt.subplots(1)
cE = Ellipse(tuple(pupil_el[:2]),
             2*pupil_el[2],
             2*pupil_el[3],
             angle=np.rad2deg(pupil_el[4]))
cL = Ellipse(tuple(iris_el[:2]),
             2*iris_el[2],
             2*iris_el[3],
             angle=np.rad2deg(iris_el[4]))

axs.imshow(I, cmap='gray')
axs.imshow(mask_epSeg, alpha=0.5)
cE.set_facecolor('None')
cE.set_edgecolor((1.0, 0.0, 0.0))
cL.set_facecolor('None')
cL.set_edgecolor((0.0, 1.0, 0.0))
axs.add_patch(cE)
axs.add_patch(cL)