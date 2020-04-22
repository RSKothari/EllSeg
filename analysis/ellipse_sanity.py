# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:41:41 2020

@author: Rudra
"""
import sys
sys.path.append('..')
import numpy as np
from helperfunctions import my_ellipse, ElliFit

H = np.array([[2/320, 0, -1], [0, 2/320, -1], [0, 0, 1]])

e1 = np.array([120, 140, 90, 120, np.deg2rad(-45)])
p1 = my_ellipse(e1)
e2 = p1.transform(H)[0][:-1]
p2 = my_ellipse(e2)
e3 = p2.transform(np.linalg.inv(H))[0][:-1]

print('Original ellipse: {}'.format(np.round(e1, 4)))
print('Normalized ellipse: {}'.format(e2))
print('Recon ellipse: {}'.format(np.round(e3, 4)))

pts1 = p1.generatePoints(50, 'random')
pts2 = p2.generatePoints(50, 'random')

fit1 = ElliFit(**{'data': np.stack(pts1, 1)})
fit2 = ElliFit(**{'data': np.stack(pts2, 1)})

print('Fit ellipse: {}'.format(fit1.model))
print('Norm fit ellipse: {}'.format(fit2.model))