#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:58:55 2020
@author: aayush
"""

import pickle
import scipy.signal as sig
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns 
import pylab
import cv2
from helperfunctions import my_ellipse
from scipy.io import loadmat
from sklearn import metrics
#%%

dataset_names=['Fuhl','LPW','NVGaze','riteyes_general','OpenEDS','PupilNet']
dataset_names=['OpenEDS','NVGaze','riteyes_general']
#dataset_names=['riteyes_general']
estimates_resized=True

def distance_metric(y_pred, y_true):
    return np.sqrt(np.sum(np.square(y_true-y_pred)))

if not estimates_resized:
    for dataset_name in dataset_names:
        filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/'+dataset_name+'_Excuse_aayush.mat'
        data_Excuse = loadmat(filename)
        data_index=(data_Excuse['filename'][0].split('.png'))
        
        if dataset_name=='OpenEDS':
            filename='/media/aaa/hdd/ALL_model/giw_e2e/op/'+dataset_name+'/ritnet_v2/_0_0opDict.pkl'
            f = open(filename, 'rb')
            data=pickle.load(f)
            f.close()
            data_gt_index=data['id']
            data_gt_pupil=data['gt']['pupil_c']
    
        if dataset_name=='NVGaze' or dataset_name=='riteyes_general':
            filename='/media/aaa/hdd/ALL_model/giw_e2e/Dataset/GT_Pupil_center_'+dataset_name+'0.pkl'        
            f = open(filename, 'rb')
            data=pickle.load(f)
            f.close()
            data_gt_index=np.array(data['Expected_filename'])
            data_gt_pupil=np.array([data['pupil_center_x'],data['pupil_center_y']]).T
        if dataset_name=='riteyes_general':
            filename='/media/aaa/hdd/ALL_model/giw_e2e/Dataset/GT_Pupil_center_riteyes_general1.pkl'        
            f = open(filename, 'rb')
            data=pickle.load(f)
            f.close()
            data_gt_index=np.array(list(data_gt_index)+\
                                   list(np.array(data['Expected_filename']).reshape(-1)))
            data_gt_pupil=np.array(list(data_gt_pupil)+list(np.array([data['pupil_center_x']\
                                   ,data['pupil_center_y']]).T))
    
        distance_save=[]
        for i in range(len(data_index)-1):
            if dataset_name=='OpenEDS':
                data_to_check=data_index[i].split('_')
                index=int(np.where(data_gt_index==np.int64(data_to_check[0]))[0])
                y_true=data_gt_pupil[index].numpy()
    
            if dataset_name=='NVGaze' or dataset_name=='riteyes_general':
    #            data_to_check=data_index[i].split('_')
    #            print (index)
    
                index=np.where(data_gt_index==data_index[i])[0]
                if np.size(index)>1:
                  index=index[0]
                y_true=data_gt_pupil[int(index)]
    
            y_pred=[data_Excuse['x'][0][i],data_Excuse['y'][0][i]]
            print (y_pred,y_true)
            distance_save.append(distance_metric(y_pred, y_true))
        print ('Saving')
        filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/'+dataset_name+'_Excuse_distance'
        np.save(filename, distance_save)

if estimates_resized:
    for dataset_name in dataset_names:
        filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/'+dataset_name+'_Excuse_aayush_resized.mat'
        data_Excuse = loadmat(filename)
        data_index=(data_Excuse['filename'][0].split('.png'))
        
        if dataset_name=='OpenEDS':
            filename='/media/aaa/hdd/ALL_model/giw_e2e/op/'+dataset_name+'/ritnet_v2/_0_0opDict.pkl'
            f = open(filename, 'rb')
            data=pickle.load(f)
            f.close()
            data_gt_index=data['id']
            data_gt_pupil=data['gt']['pupil_c']
    
        if dataset_name=='NVGaze' or dataset_name=='riteyes_general':
            filename='/media/aaa/hdd/ALL_model/giw_e2e/Dataset/GT_Pupil_center_'+dataset_name+'0.pkl'        
            f = open(filename, 'rb')
            data=pickle.load(f)
            f.close()
            data_gt_index=np.array(data['Expected_filename'])
            data_gt_pupil=np.array([data['pupil_center_x'],data['pupil_center_y']]).T
        if dataset_name=='riteyes_general':
            filename='/media/aaa/hdd/ALL_model/giw_e2e/Dataset/GT_Pupil_center_riteyes_general1.pkl'        
            f = open(filename, 'rb')
            data=pickle.load(f)
            f.close()
            data_gt_index=np.array(list(data_gt_index)+\
                                   list(np.array(data['Expected_filename']).reshape(-1)))
            data_gt_pupil=np.array(list(data_gt_pupil)+list(np.array([data['pupil_center_x']\
                                   ,data['pupil_center_y']]).T))
    
        distance_save=[]
        for i in range(len(data_index)-1):
            if dataset_name=='OpenEDS':
                data_to_check=data_index[i].split('_')
                index=int(np.where(data_gt_index==np.int64(data_to_check[0]))[0])
                y_true=data_gt_pupil[index].numpy()
    
            if dataset_name=='NVGaze' or dataset_name=='riteyes_general':
    #            data_to_check=data_index[i].split('_')
    #            print (index)
    
                index=np.where(data_gt_index==data_index[i])[0]
                if np.size(index)>1:
                  index=index[0]
                y_true=data_gt_pupil[int(index)]
    
            y_pred=[data_Excuse['x'][0][i],data_Excuse['y'][0][i]]
            print (y_pred,y_true)
            distance_save.append(distance_metric(y_pred/np.array([1.2]), y_true))
        print ('Saving')
        filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/'+dataset_name+'_Excuse_distance_resized'
        np.save(filename, distance_save)
