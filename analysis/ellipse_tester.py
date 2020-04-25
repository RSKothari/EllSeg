#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:38:04 2019

@author: rakshit
"""

import sys

sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from helperfunctions import my_ellipse
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.draw import ellipse

def getPts(param):
    ellipse_mod = my_ellipse(param)
    x_1, y_1 = ellipse_mod.generatePoints(50, 'equiSlope')
    x_2, y_2 = ellipse_mod.generatePoints(50, 'equiAngle')
    return np.concatenate([x_1, x_2]), np.concatenate([y_1, y_2])

def sketch(ptNum, data):
    x = np.round(data['xPts'][:, ptNum]).astype(np.int)
    y = np.round(data['yPts'][:, ptNum]).astype(np.int)
    patches = []
    for k in range(len(x)):
        patch = data['I'][k, (y[k]-20):(y[k]+20), (x[k]-20):(x[k]+20)]
        patches.append(patch)
    return patches

n_pts = 5
I_res = (240, 240)
H = np.array([[2/240, 0, -1], [0, 2/240, -1], [0, 0, 1]])
angleItr = np.linspace(10, 170, n_pts)
eccItr = np.linspace(0.4, 1.6, n_pts) + 1e-5 # Never a perfect 1

data = {'I':[], 'xPts':[], 'yPts':[]}

for ang in np.nditer(angleItr):
    for ecc in np.nditer(eccItr):
        # Input as param
        param = np.array([130, 110, 40, 40*ecc, np.deg2rad(ang)])
        #print('Phi: {}'.format(np.round(my_ellipse(param).recover_Phi(), 2)))
        param_norm = my_ellipse(param).transform(H)[0][:5]
        print('Theta: {}. E: {}'.format(ang, ecc))
        print('Param: {}'.format(np.round(param, 2)))
        print('Param norm: {}'.format(np.round(param_norm, 2)))
        print('Phi: {}'.format(np.round(my_ellipse(param_norm).recover_Phi(), 2)))
        x_pts, y_pts = getPts(param)
        I = np.zeros(I_res)
        rr, cc = ellipse(param[1], param[0], param[3], param[2], shape=I_res, rotation=param[4])
        I[rr, cc] = 1
        data['I'].append(I)
        data['xPts'].append(x_pts)
        data['yPts'].append(y_pts)

data['I'] = np.stack(data['I'], axis=0)
data['xPts'] = np.stack(data['xPts'], axis=0)
data['yPts'] = np.stack(data['yPts'], axis=0)

I = sketch(9, data)
fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(n_pts, n_pts), axes_pad=0.6)

for i, ax in enumerate(grid):
    ax.imshow(I[i])
    ax.scatter(20, 20)
    ang = angleItr[int(np.floor(i/n_pts))]
    ecc = np.round(eccItr[int(i%n_pts)], 3)
    ax.set_title('Orient: {}.\nEcc: {}'.format(ang, ecc))

plt.show()

'''
param = [220, 260, 80, 50, np.deg2rad(15)]
ellipse_mod = my_ellipse(param)
param_test_mat = ellipse_mod.mat2param(ellipse_mod.mat)[:-1]
param_test_quad = ellipse_mod.quad2param(ellipse_mod.quad)[:-1]

test1 = np.sum(np.abs(param - param_test_quad)) < 1e-2
test2 = np.sum(np.abs(param - param_test_mat)) < 1e-2
print('Original param: {}'.format(bool(test1)))
print('Test: param to quad: {}'.format(bool(test2)))

x_pts, y_pts = ellipse_mod.generatePoints(50, 'equiAngle')

I = np.zeros((640, 480))
rr, cc = ellipse(param[1], param[0], param[3], param[2], shape=(480, 640), rotation=param[4])
I[rr, cc] = 1

plt.imshow(I)
plt.scatter(x_pts, y_pts)

'''