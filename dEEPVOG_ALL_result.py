#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:00:53 2020

@author: aayush
See DeepVOG accuracy
First part is segmentation accuracy based on original network
Second part loads the predicted accuracy by the segmentation model of deepvog and 
outputs the pkl file which contains all the pixel error
"""

#from DeepVOG_model import load_DeepVOG
import skimage.io as ski
import numpy as np
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

save_pupil_center_error=True

def distance_metric(y_pred, y_true):
    return np.sqrt(np.sum(np.square(y_true-y_pred)))


cond_name='OpenEDS'
save_no=''
filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
f = open(filename, 'rb')
data=pickle.load(f)
f.close()
print ('Average mIoU for'+cond_name+ 'is',np.average(data['average_mIoU']))
    
cond_name='NVGaze'
save_no=''
filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
f = open(filename, 'rb')
data=pickle.load(f)
f.close()
print ('Average mIoU for'+cond_name+ 'is',np.average(data['average_mIoU']))


cond_name='riteyes_general'
save_no='0'
data_miou=[]
filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
f = open(filename, 'rb')
data=pickle.load(f)
f.close()
data_miou=(data['average_mIoU'])
save_no='1'
filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
f = open(filename, 'rb')
data=pickle.load(f)
f.close()
data_miou+=data['average_mIoU']
save_no='2'
filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
f = open(filename, 'rb')
data=pickle.load(f)
f.close()
data_miou+=data['average_mIoU']
print ('Average mIoU for'+cond_name+ 'is',np.average(data_miou))


cond_name='OpenEDS'
save_no=''
filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
f = open(filename, 'rb')
data=pickle.load(f)
f.close()

from string import punctuation
pathData='/media/aaa/hdd/ALL_model/giw_e2e/deepvog_original_model/OpenEDS.txt'
fileDescriptor = open(pathData, "r")
line = True
Images_with_Labels=[]
pupil_x=[]
pupil_y=[]
string_bracket=list(set(punctuation))[6]
while line:
    line = fileDescriptor.readline()
    if 'Fit OpenEDS.mp4' in line:
        if len(line.split('/'))==3:
            pupil_x.append(np.float(line.split('/')[1].split(string_bracket)[1]))
            pupil_y.append(np.float(line.split('/')[2].strip('\n')))
        else:
            pupil_x.append(0)
            pupil_y.append(0)
  
    
pupil_distance=[]
for i in range(2376-1):
    pred_pupil=[pupil_x[i+1],pupil_y[i+1]]
    gt_pupil=[data['pupil_center_x'][i].numpy(),data['pupil_center_y'][i].numpy()]
    if pred_pupil[0]!=0:
        pupil_distance.append(distance_metric(np.array(pred_pupil),np.array(gt_pupil)))
filename='/media/aaa/hdd/ALL_model/DeepVOG_pretrained_pupil_center_error'
np.save(filename, pupil_distance)