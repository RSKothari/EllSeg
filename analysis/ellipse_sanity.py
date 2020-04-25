# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:41:41 2020

@author: Rudra
"""
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

from helperfunctions import my_ellipse, ElliFit
from matplotlib.patches import Ellipse

def verifyMat(M, pts):
    dist = 0
    for i in range(pts.shape[0]):
        pt = pts[i, :]
        pt = pt[:, np.newaxis]
        dist += pt.T.dot(M).dot(pt)
    return dist

cx = 120
cy = 140
a = 90
b = 120
deg = 30

H = np.array([[2/320, 0, -1], [0, 2/320, -1], [0, 0, 1]])

e1 = np.array([cx, cy, a, b, np.deg2rad(deg)])
p1 = my_ellipse(e1)
e2 = p1.transform(H)[0][:-1]
p2 = my_ellipse(e2)
e3 = p2.transform(np.linalg.inv(H))[0][:-1]

print('Original ellipse: {}'.format(np.round(e1, 4)))
print('Normalized ellipse: {}'.format(e2))
print('Recon ellipse: {}'.format(np.round(e3, 4)))

pts1 = p1.generatePoints(50, 'equiAngle')
pts2 = p2.generatePoints(50, 'equiAngle')
N = len(pts1[0])

fit1 = ElliFit(**{'data': np.stack(pts1, 1)})
fit2 = ElliFit(**{'data': np.stack(pts2, 1)})

print('Fit ellipse: {}'.format(fit1.model))
print('Norm fit ellipse: {}'.format(fit2.model))

el1 = Ellipse((cx, cy), 2*a, 2*b, angle=deg)
fig, ax = plt.subplots(1)
ax.scatter(pts1[0], pts1[1])
ax.add_patch(el1)
plt.show(block=False)

# Points to matrix fit verification
P = np.stack(pts1 + (np.ones(N, ), ), axis=1)
dist = verifyMat(p1.mat, P)

