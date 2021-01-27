#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:51:29 2021

@author: rakshit
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

path_images = '/media/rakshit/tank/Dataset/Fuhl/data set XIX'
list_images = glob.glob(path_images + '/*.png')

im_num = [int(os.path.splitext(os.path.split(ele)[1])[0]) for ele in list_images]
im_num = np.sort(np.array(im_num))

loc = np.where(np.diff(im_num) > 1)[0] + 1

# 'ffmpeg -r 120 -f image2 -start_number '