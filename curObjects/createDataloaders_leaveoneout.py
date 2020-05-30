#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

@author: rakshit
'''
import os
import sys
import copy
import pickle

sys.path.append('..')
import CurriculumLib as CurLib
from CurriculumLib import DataLoader_riteyes

path2data = '/media/rakshit/tank/Dataset'
path2h5 = os.path.join(path2data, 'All')
keepOld = True

DS_sel = pickle.load(open('dataset_selections.pkl', 'rb'))
AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
list_ds = list(DS_sel['train'].keys())

testObj = []
for selSet_out in list_ds:
    print('--------------------')
    print('Removing {}'.format(selSet_out))
    
    list_notOut = copy.deepcopy(list_ds)
    list_notOut.remove(selSet_out)
    
    # Select training subsets from all datasets except <selSet_out>
    subsets_train = []
    for selSet in list_notOut:
        subsets_train += DS_sel['train'][selSet]
        
    # Train object
    AllDS_cond = CurLib.selSubset(AllDS, subsets_train)
    dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='vanilla', notest=True)
    trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), scale=0.5)
    validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), scale=0.5)
    
    path2save = os.path.join(os.getcwd(), 'leaveoneout', 'cond_'+selSet_out+'.pkl')
    
    if os.path.exists(path2save) and keepOld:
        print('Preserving old selections ...')
        # This ensure that the original selection remains the same
        trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
        trainObj.imList = trainObj_orig.imList
        validObj.imList = validObj_orig.imList
        pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))
    else:
        pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))