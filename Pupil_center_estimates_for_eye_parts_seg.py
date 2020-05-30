#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:27:33 2020

@author: aaa
"""

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
from helperfunctions import generateEmptyStorage, getValidPoints
from helperfunctions import ransac, ElliFit, my_ellipse
#%%

#dataset_names=['Fuhl','LPW','NVGaze','riteyes_general','OpenEDS','PupilNet']
#dataset_names=['OpenEDS','NVGaze','riteyes_general']#'riteyes_general']
dataset_names=['riteyes_general']
#model_names=['ritnet_v4']#,'ritnet_v5']#,'ritnet_v3']#,'deepvog']
model_names=['ritnet_v5']
test_cases='_0_0'#,'_1_0']
#test_cases=''#,'_1_0']

#folder_name='GIW_e2e_temp'
folder_name='giw_e2e'

def distance_metric(y_pred, y_true):
    return np.sqrt(np.sum(np.square(y_true-y_pred)))
  #%%
def get_iris_pupil_center_from_eye_parts(data_to_fit):
    pupilPts, irisPts = getValidPoints(data_to_fit)
    # Pupil ellipse fit
#    print (pupilPts,irisPts)
    if len (pupilPts)>0:
        model_pupil = ransac(pupilPts, ElliFit, 15, 40, 5e-3, 15).loop()
        pupil_fit_error = my_ellipse(model_pupil.model).verify(pupilPts)
        
        r, c = np.where(data_to_fit == 2)
        pupil_loc = model_pupil.model[:2] if pupil_fit_error < 0.05 else np.stack([np.mean(c), np.mean(r)], axis=0)
    else:
        pupil_loc=None
    # Iris ellipse fit
    if len (irisPts)>0:
    
        model_iris = ransac(irisPts, ElliFit, 15, 40, 5e-3, 15).loop()
        iris_fit_error = my_ellipse(model_iris.model).verify(irisPts)
        iris_loc=model_iris.model[0:2]  
    else:
        iris_loc=None
    return pupil_loc,iris_loc
#%%
for dataset_name in dataset_names:
    
    for model_name in model_names:
        save_filename=[]
        pupil_distance=[]
        iris_distance=[]
        invalid_counter_iris=0
        invalid_counter_pupil=0

        filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
                model_name+'/'+test_cases+'opDict.pkl'
        f = open(filename, 'rb')
        data=pickle.load(f)
        f.close()
        for i in range(len(data['gt']['pupil_c'])):
            pred=data['pred']['mask'][i]
            gt_pupil=data['gt']['pupil_c'][i]
            gt_iris=data['gt']['iris_c'][i]
            if (folder_name=='giw_e2e') & (model_name!='deepvog'):
                pred=pred+1
            if (model_name=='deepvog'):
                pred=pred+2
            pred_pupil,pred_iris=get_iris_pupil_center_from_eye_parts(pred)
            if pred_pupil is None:
                invalid_counter_pupil+=1
            else:
                pupil_distance.append(distance_metric(pred_pupil, gt_pupil.numpy()))
            if pred_iris is None:
                invalid_counter_iris+=1
            else:
                iris_distance.append(distance_metric(pred_iris, gt_iris.numpy()))
#            print (i,pred_pupil,pred_iris)

        filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
        model_name+'/opDict_pupil_center_estimate_eye_parts'+str(invalid_counter_pupil)
        np.save(filename, pupil_distance)
        filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
        model_name+'/opDict_iris_center_estimate_eye_parts'+str(invalid_counter_iris)
        np.save(filename, iris_distance)
#       

