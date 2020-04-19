# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:41:41 2020

@author: Rudra
"""
import sys
sys.path.append('..')
import numpy as np
from helperfunctions import my_ellipse

H = np.array([[2/320, 0, -1], [0, 2/240, -1], [0, 0, 1]])

e1 = np.array([120, 140, 90, 120, np.deg2rad(-80)])
p1 = my_ellipse(e1)
e2 = p1.transform(H)[0][:-1]
p2 = my_ellipse(e2)
e3 = p2.transform(np.linalg.inv(H))[0][:-1]

print('Original ellipse: {}'.format(np.round(e1, 4)))
print('Recon ellipse: {}'.format(np.round(e3, 4)))
