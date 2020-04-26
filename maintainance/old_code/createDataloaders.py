#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

@author: rakshit
'''
import os
import pickle
import CurriculumLib as CurLib

from CurriculumLib import DataLoader_riteyes

path2data = '/media/rakshit/tank/Dataset'
path2h5 = os.path.join(path2data, 'All')
keepOld = True

#%% Train on NVGaze and OpenEDS
datasets = ['NVGaze', 'OpenEDS']
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
subsets = nv_subs1 + nv_subs2 + ['train']

AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
AllDS = CurLib.selDataset(AllDS, datasets)
AllDS = CurLib.selSubset(AllDS, subsets)
datasets_present, subsets_present = CurLib.listDatasets(AllDS)
print('Datasets selected ---------')
print(datasets_present)
print('Subsets selected ---------')
print(subsets_present)

dataDiv_obj = CurLib.generate_fileList(AllDS, mode = 'vanilla', notest=True)
trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), scale=0.5)
validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), scale=0.5)

# Test on OpenEDS, NVGaze
datasets = ['NVGaze', 'OpenEDS']
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 4)]
subsets = nv_subs1 + nv_subs2 + ['validation']

AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
AllDS = CurLib.selDataset(AllDS, datasets)
AllDS = CurLib.selSubset(AllDS, subsets)
datasets_present, subsets_present = CurLib.listDatasets(AllDS)
print('Datasets selected ---------')
print(datasets_present)
print('Subsets selected ---------')
print(subsets_present)

dataDiv_obj = CurLib.generate_fileList(AllDS, mode = 'none', notest=True)
testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'test', False, (480, 640), sort='ordered', scale=0.5)

#
path2save = os.path.join(os.getcwd(), 'curObjects', 'cond_0.pkl')
if os.path.exists(path2save) and keepOld:
    print('Preserving old selections ...')
    # This ensure that the original selection remains the same
    trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
    trainObj.imList = trainObj_orig.imList
    validObj.imList = validObj_orig.imList
    testObj.imList = testObj_orig.imList
pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))

#%% Train on OpenEDS, NVGaze, LPW
datasets = ['NVGaze', 'OpenEDS', 'LPW']
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
lpw_subs = ['LPW_{}'.format(i+1) for i in range(0, 16)]
subsets = nv_subs1 + nv_subs2 + lpw_subs + ['train']

AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
AllDS = CurLib.selDataset(AllDS, datasets)
AllDS = CurLib.selSubset(AllDS, subsets)
datasets_present, subsets_present = CurLib.listDatasets(AllDS)
print('Datasets selected ---------')
print(datasets_present)
print('Subsets selected ---------')
print(subsets_present)

dataDiv_obj = CurLib.generate_fileList(AllDS, mode = 'vanilla', notest=True)
trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), scale=0.5)
validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), scale=0.5)

# Test on OpenEDS, NVGaze, LPW
datasets = ['NVGaze', 'OpenEDS', 'LPW']
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 2)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 2)]
lpw_subs = ['LPW_{}'.format(i+1) for i in range(16, 22)]
subsets = nv_subs1 + nv_subs2 + lpw_subs + ['validation']

AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
AllDS = CurLib.selDataset(AllDS, datasets)
AllDS = CurLib.selSubset(AllDS, subsets)
datasets_present, subsets_present = CurLib.listDatasets(AllDS)
print('Datasets selected ---------')
print(datasets_present)
print('Subsets selected ---------')
print(subsets_present)

dataDiv_obj = CurLib.generate_fileList(AllDS, mode = 'none', notest=True)
testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'test', False, (480, 640), sort='ordered', scale=0.5)

#
path2save = os.path.join(os.getcwd(), 'curObjects', 'cond_1.pkl')
if os.path.exists(path2save) and keepOld:
    print('Preserving old selections ...')
    # This ensure that the original selection remains the same
    trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
    trainObj.imList = trainObj_orig.imList
    validObj.imList = validObj_orig.imList
    testObj.imList = testObj_orig.imList
pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))

#%% Train on OpenEDS, NVGaze, LPW, S-General
datasets = ['NVGaze', 'OpenEDS', 'LPW', 'riteyes_general']
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
lpw_subs = ['LPW_{}'.format(i+1) for i in range(0, 16)]
riteyes_subs = ['riteyes_general_{}'.format(i+1) for i in range(0, 18)]
subsets = nv_subs1 + nv_subs2 + lpw_subs + ['train'] + riteyes_subs

AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
AllDS = CurLib.selDataset(AllDS, datasets)
AllDS = CurLib.selSubset(AllDS, subsets)
datasets_present, subsets_present = CurLib.listDatasets(AllDS)
print('Datasets selected ---------')
print(datasets_present)
print('Subsets selected ---------')
print(subsets_present)

dataDiv_obj = CurLib.generate_fileList(AllDS, mode = 'vanilla', notest=True)
trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), scale=0.5)
validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), scale=0.5)

# Test on OpenEDS, NVGaze, LPW, S-General
datasets = ['NVGaze', 'OpenEDS', 'LPW', 'riteyes_general']
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 2)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 2)]
lpw_subs = ['LPW_{}'.format(i+1) for i in range(16, 22)]
riteyes_subs = ['riteyes_general_{}'.format(i+1) for i in range(18, 24)]
subsets = nv_subs1 + nv_subs2 + lpw_subs + ['validation'] + riteyes_subs

AllDS = CurLib.readArchives(os.path.join(path2data, 'MasterKey'))
AllDS = CurLib.selDataset(AllDS, datasets)
AllDS = CurLib.selSubset(AllDS, subsets)
datasets_present, subsets_present = CurLib.listDatasets(AllDS)
print('Datasets selected ---------')
print(datasets_present)
print('Subsets selected ---------')
print(subsets_present)

dataDiv_obj = CurLib.generate_fileList(AllDS, mode = 'none', notest=True)
testObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'test', False, (480, 640), sort='ordered', scale=0.5)

#
path2save = os.path.join(os.getcwd(), 'curObjects', 'cond_2.pkl')
if os.path.exists(path2save) and keepOld:
    print('Preserving old selections ...')
    # This ensure that the original selection remains the same
    trainObj_orig, validObj_orig, testObj_orig = pickle.load(open(path2save, 'rb'))
    trainObj.imList = trainObj_orig.imList
    validObj.imList = validObj_orig.imList
    testObj.imList = testObj_orig.imList
pickle.dump((trainObj, validObj, testObj), open(path2save, 'wb'))

