#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:31:51 2020

@author: aaa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:08:51 2020

@author: rakshit
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from helperfunctions import mypause
from torch.utils.data import DataLoader
from utils import generateImageGrid, points_to_heatmap
from CurriculumLib import readArchives, listDatasets, generate_fileList
from CurriculumLib import selDataset, selSubset, DataLoader_riteyes
import subprocess

if __name__=='__main__':
    path2data = '/media/aaa/hdd/ALL_model/giw_e2e/Dataset'
    path2h5 = os.path.join(path2data, 'All')
    path2arc_keys = os.path.join(path2data, 'MasterKey')
    '''
    AllDS = readArchives(path2arc_keys)
    datasets_present, subsets_present = listDatasets(AllDS)
    print('Datasets present -----')
    print(datasets_present)
    print('Subsets present -----')
    print(subsets_present)

    nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 5) for j in range(0, 3)]
    nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 5) for j in range(0, 3)]
    lpw_subs = ['LPW_{}'.format(i+1) for i in range(0, 12)]
    subsets = nv_subs1 + nv_subs2 + lpw_subs + ['none', 'train']

    AllDS = selDataset(AllDS, ['OpenEDS', 'UnityEyes', 'NVGaze', 'LPW', 'riteyes_general'])
    AllDS = selSubset(AllDS, subsets)
    dataDiv_obj = generate_fileList(AllDS, mode='vanilla', notest=True)
    trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), 0.5)
    validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), 0.5)
    '''
    
    cond_name='OpenEDS'
    f = os.path.join('curObjects','baseline', 'cond_'+cond_name+'.pkl')
    trainObj, validObj, testObj = pickle.load(open(f, 'rb'))
    trainObj.path2data = path2h5
    testObj.path2data = path2h5
    validObj.path2data = path2h5

#%%
    trainLoader = DataLoader(testObj,
                             batch_size=1,
                             shuffle=True,
                             num_workers=8,
                             drop_last=True)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    totTime = []
    startTime = time.time()
    opDict = {'Expected_filename':[], 'pupil_center_x':[],'pupil_center_y':[]}
    Expected_filename=[]
    pupil_center_x=[]
    pupil_center_y=[]
    counter=0
    for bt, data in enumerate(trainLoader):
        print (bt)
#        I, mask, spatialWeights, distMap, pupil_center, iris_center, elPts, elNorm, cond, imInfo = data
        I, mask, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = data
        
        ##change curriculum file
        for i in range (I.shape[0]):
          #saving Images
#            print (i)
#            img=Image.fromarray(I[i].numpy())
            
#            mask_current=mask[i].numpy()
#            image_current=np.uint8(I[i].numpy())
#            mask_current[mask_current>1]=1
#            image_current=image_current*mask_current
#            image_current[np.where(mask_current==np.int32(0))]=127
#
#            img=Image.fromarray(np.uint8(image_current))

#            img.save('/media/aaa/hdd/ALL_model/giw_e2e/Dataset/testdataset/riteyes_general/'+str(int(imInfo[i,0]))+\
#                     '_'+str(int(pupil_center[i][0]))+'_'+str(int(pupil_center[i][1]))+'.png')
            
          
#          Ssaving GT
            Expected_filename.append(str(int(imInfo[i,0]))+\
                     '_'+str(int(pupil_center[i][0]))+'_'+str(int(pupil_center[i][1])))
            pupil_center_x.append(pupil_center[i][0].numpy())
            pupil_center_y.append(pupil_center[i][1].numpy())
            
            if len(pupil_center_x)==6000:
                print ('here')
                opDict['Expected_filename']=(Expected_filename)
                opDict['pupil_center_x']=(pupil_center_x)
                opDict['pupil_center_y']=(pupil_center_y)
                Expected_filename=[]
                pupil_center_x=[]
                pupil_center_y=[]
                
                filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/GT_Pupil_center_'+cond_name+str(counter)+'.pkl'              
                print('--- Saving output directory ---')
                f = open(filename, 'wb')
                pickle.dump(opDict, f)
                f.close()
                counter+=1
                opDict = {'Expected_filename':[], 'pupil_center_x':[],'pupil_center_y':[]}                
    print ('here')
    opDict['Expected_filename']=(Expected_filename)
    opDict['pupil_center_x']=(pupil_center_x)
    opDict['pupil_center_y']=(pupil_center_y)
    Expected_filename=[]
    pupil_center_x=[]
    pupil_center_y=[]
    
    filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/GT_Pupil_center_'+cond_name+str(counter)+'.pkl'              
    print('--- Saving output directory ---')
    f = open(filename, 'wb')
    pickle.dump(opDict, f)
    f.close()
    counter+=1
    opDict = {'Expected_filename':[], 'pupil_center_x':[],'pupil_center_y':[]}                
 
    print('Time for 1 epoch: {}'.format(np.sum(totTime)))
