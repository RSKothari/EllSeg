#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

@author: rakshit
'''
import os
import sys
import pickle

sys.path.append('..')
import CurriculumLib as CurLib
from CurriculumLib import DataLoader_riteyes

path2data = '/media/rakshit/tank/Dataset'
path2h5 = os.path.join(path2data, 'All')
keepOld = True

DS_sel = pickle.load(open('dataset_selections.pkl', 'rb'))
AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
list_ds = ['NVGaze', 'OpenEDS', 'riteyes_general', 'LPW', 'Fuhl', 'PupilNet']

# Generate objects per dataset
for setSel in list_ds:

    # Train and test object
    AllDS_cond = CurLib.selDataset(AllDS, setSel)
    dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='vanilla', notest=False)
    trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), scale=0.5)
    validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), scale=0.5)
    testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'test', False, (480, 640), scale=0.5)
    
    path2save = os.path.join(os.getcwd(), 'random', 'cond_'+setSel+'.pkl')
    if os.path.exists(path2save) and keepOld:
        print('Preserving old selections ...')
        # This ensure that the original selection remains the same
        trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
        trainObj.imList = trainObj_orig.imList
        validObj.imList = validObj_orig.imList
        testObj.imList = testObj_orig.imList
        pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
    else:
        pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))