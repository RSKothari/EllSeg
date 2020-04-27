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
for train_set in list_ds:

    # Train object
    AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train'][train_set])
    dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='vanilla', notest=False)
    trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), scale=0.5)
    validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), scale=0.5)

    # Test object
    AllDS_cond = CurLib.selSubset(AllDS, DS_sel['test'][train_set])
    dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='none', notest=True)
    testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'test', False, (480, 640), scale=0.5)

    path2save = os.path.join(os.getcwd(), 'baseline', 'cond_'+train_set+'.pkl')
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