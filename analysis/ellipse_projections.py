# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 07:25:40 2020

@author: Rudra
"""
import sys
sys.path.append('..')

import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

from helperfunctions import my_ellipse, mypause
from matplotlib.patches import Ellipse
from skimage.draw import ellipse

deg_list = np.linspace(0, np.pi, 400)

res = (500, 500)
cx = 260
cy = 220
a = 150
b = 90

I_z = np.random.rand(*res)

'''
# Ellipse: Negate the sign unless you're using ElliFit to derive the Ellipse.
# Image coordinates Y axis goes negative upwards, hence invert rotation
el = Ellipse((cx, cy), 2*a, 2*b, -45.0)
el.set_facecolor('None')
el.set_edgecolor((1.0, 0.0, 0.0))


# Observe integral change
fig, axs = plt.subplots(2,2)
I_obj = axs[0,0].imshow(I_z, cmap='gray')
axs[0,0].add_patch(el)
pObj_x = axs[0,1].plot(np.arange(res[0]), 500*np.random.rand(res[0]))[0]
pObj_y = axs[1,0].plot(np.arange(res[0]), 500*np.random.rand(res[1]))[0]

plt.pause(.1)
plt.draw()

for deg in np.nditer(deg_list):
    print('Degree: {}'.format(deg))
    el_model = np.array([cx, cy, a, b, deg])

    rr, cc = ellipse(cy, cx, b, a, rotation=deg.item())
    proj = copy.deepcopy(I_z)
    proj[rr, cc] = 1.0
    proj_x, proj_y = np.sum(proj, axis=0), np.sum(proj, axis=1)

    # Update figure objects
    el.angle = -np.rad2deg(deg)
    I_obj.set_data(proj)
    pObj_x.set_data(np.arange(res[0]), proj_x)
    pObj_y.set_data(np.arange(res[0]), proj_y)
    plt.pause(0.1)
    plt.draw()
    plt.show(block=False)
'''
# Observe magnitude
el = Ellipse((cx, cy), 2*a, 2*b, -45.0)
el.set_facecolor('None')
el.set_edgecolor((1.0, 0.0, 0.0))

fig, axs = plt.subplots(2,2)
I_obj = axs[0,0].imshow(I_z)
axs[0,0].add_patch(el)

pObj_x = axs[0,1].plot(np.arange(res[0]), 1000*(np.random.rand(res[0])-0.5))[0]
pObj_y = axs[1,0].plot(np.arange(res[0]), 1000*(np.random.rand(res[1])-0.5))[0]

plt.pause(.1)
plt.draw()

for deg in np.nditer(deg_list):
    print('Degree: {}'.format(deg))
    el_model = np.array([cx, cy, a, b, deg])

    rr, cc = ellipse(cy, cx, b, a, rotation=deg.item())
    proj = copy.deepcopy(I_z)
    proj[rr, cc] = 1
    proj_x, proj_y = np.sum(proj, axis=0), np.sum(proj, axis=1)

    sobelx = cv2.Sobel(proj, cv2.CV_64F, 1, 0, ksize=31)
    sobely = cv2.Sobel(proj, cv2.CV_64F, 0, 1, ksize=31)
    mag = (sobelx**2 + sobely**2)**0.5
    mag = mag/mag.max()
    ang = np.arctan((sobely+1e-7)/(sobelx+1e-7))
    score = mag*ang
    proj_x, proj_y = np.sum(score, axis=0), np.sum(score, axis=1)

    # Update figure objects
    I_obj.set_data(proj)
    el.angle = -np.rad2deg(deg)
    pObj_x.set_data(np.arange(res[0]), proj_x)
    pObj_y.set_data(np.arange(res[0]), proj_y)
    plt.pause(0.1)
    plt.draw()
    plt.show(block=False)