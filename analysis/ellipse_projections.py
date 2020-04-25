# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 07:25:40 2020

@author: Rudra
"""
import sys
sys.path.append('..')

import copy
import numpy as np
import matplotlib.pyplot as plt

from helperfunctions import my_ellipse, mypause
from matplotlib.patches import Ellipse
from skimage.draw import ellipse

deg_list = np.linspace(-np.pi, np.pi, 500)

res = (500, 500)
cx = 260
cy = 220
a = 150
b = 90

el = Ellipse((cx, cy), 2*a, 2*b, 45.0)
I_z = np.zeros(res)

axs = plt.subplot(111, aspect='equal')
el.set_clip_box(axs.bbox)
el.set_alpha(0.1)
axs.add_artist(el)

plt.xlim(0, 700)
plt.ylim(0, 700)

#pObj_x = axs[0,1].plot(np.zeros(res[0]))[0]
#pObj_y = axs[1,0].plot(np.zeros(res[1]))[0]
#plt.pause(.01)
#plt.draw()
'''
for deg in np.nditer(deg_list):
    print('Degree: {}'.format(deg))
    #el_model = np.array([cx, cy, a, b, deg])
    rr, cc = ellipse(cy, cx, b, a, rotation=deg.item())
    proj = copy.deepcopy(I_z)
    proj[rr, cc] = 1
    proj_x, proj_y = np.sum(proj, axis=0), np.sum(proj, axis=1)

    # Update figure objects
    el.angle = -np.rad2deg(deg)
    pObj_x.set_ydata = proj_x
    pObj_y.set_ydata = proj_y
    mypause(0.01)
'''