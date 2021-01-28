#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:06:33 2019

@author: rakshit
"""

import re
import os
import cv2
import pdb
import h5py
import copy
import torch

import numpy as np
import scipy.io as scio

from data_augment import augment
from torch.utils.data import Dataset

from helperfunctions import simple_string, one_hot2dist, extract_datasets
from helperfunctions import my_ellipse, pad2Size, get_ellipse_info

from sklearn.model_selection import StratifiedKFold, train_test_split

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Deactive file locking

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class DataLoader_riteyes(Dataset):
    def __init__(self, dataDiv_Obj, path2data, fold_num, cond, augFlag, size, sort='random', scale=False):

        cond = 'train_idx' if 'train' in cond else cond
        cond = 'valid_idx' if 'valid' in cond else cond
        cond = 'test_idx' if 'test' in cond else cond

        # Operational variables
        self.scale = scale
        self.augFlag = augFlag
        self.imList = dataDiv_Obj.folds[fold_num][cond]
        self.arch = dataDiv_Obj.arch
        self.path2data = path2data
        self.size = size
        self.sort(sort)
        self.prec = torch.float32

        # Rank datasets by archive ID
        #dsnums = np.unique(self.imList[:, 1], return_inverse=True)[1]
        dsnums = extract_datasets(self.arch[self.imList[:, 1]])[1] # Each entry will be mapped to a dataset ID
        self.imList = np.hstack([self.imList, dsnums[:, np.newaxis]])

    def sort(self, sort):

        if sort=='ordered':
            # Completely ordered
            loc = np.unique(self.imList,
                            return_counts=True,
                            axis=0)
            print('Warning. Non-unique file list.') if np.any(loc[1]!=1) else print('Sorted list')
            self.imList = loc[0]

        elif sort=='semiordered':
            # Randomize first, then sort by archNum
            loc = np.random.permutation(self.imList.shape[0])
            self.imList = self.imList[loc, :]
            loc = np.argsort(self.imList[:, 1])
            self.imList = self.imList[loc, :]

        elif sort=='random':
            # Completely random selection. DEFAULT.
            loc = np.random.permutation(self.imList.shape[0])
            self.imList = self.imList[loc, :]

    def scaleFn(self, img, label, elParam, pupil_center):
        dsize = (int(self.scale*img.shape[1]), int(self.scale*img.shape[0]))
        H = np.array([[self.scale, 0, 0],
                      [0, self.scale, 0],
                      [0, 0, 1]])
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
        label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
        elParam_1 = my_ellipse(elParam[0]).transform(H)[0][:-1] if not np.all(elParam[0]==-1) else elParam[0]
        elParam_2 = my_ellipse(elParam[1]).transform(H)[0][:-1] if not np.all(elParam[0]==-1) else elParam[0]
        elParam = (elParam_1, elParam_2)
        pupil_center = H[:2, :2].dot(pupil_center) if not np.all(pupil_center==-1) else pupil_center
        return img, label, elParam, pupil_center

    def __len__(self):
        return self.imList.shape[0]

    def __getitem__(self, idx):
        '''
        Reads in an image and all the required sources of information.
        Also returns a flag tensor where a 0 in:
            pos 0: indicates pupil center exists
            pos 1: indicates mask exists
            pos 2: indicates pupil ellipse exists
            pos 3: indicates iris ellipse exists
            ##modified:
        '''
        numClasses = 3
        img, label, elParam, pupil_center, cond, imInfo = self.readImage(idx)
        img, label, pupil_center, elParam = pad2Size(img,
                                                    label,
                                                    elParam,
                                                    pupil_center,
                                                    self.size)

        if self.scale:
            img, label, elParam, pupil_center = self.scaleFn(img, label, elParam, pupil_center)
        img, label, pupil_center, elParam = augment(img,
                                                    label,
                                                    pupil_center,
                                                    elParam) if self.augFlag else (img,
                                                                                   label,
                                                                                   pupil_center,
                                                                                   elParam)

        # Modify labels by removing Sclera class
        label[label == 1] = 0 # If Sclera exists, move it to background.
        label[label == 2] = 1 # Move Iris to 1
        label[label == 3] = 2 # Move Pupil to 2

        # Compute edge weight maps
        spatialWeights = cv2.Canny(label.astype(np.uint8), 0, 1)/255
        spatialWeights = 1 + cv2.dilate(spatialWeights,(3,3), iterations = 1)*20

        # Calculate distMaps for only Iris and Pupil. Pupil: 2. Iris: 1. Rest: 0.
        distMap = np.zeros((3, *img.shape))

        # Find distance map for each class
        for i in range(0, numClasses):
            distMap[i, ...] = one_hot2dist(label.astype(np.uint8)==i)

        # Convert data to torch primitives
        img = (img - img.mean())/img.std()
        img = torch.from_numpy(img).unsqueeze(0).to(self.prec) # Adds a singleton for channels

        # Groundtruth annotation
        label = MaskToTensor()(label).to(torch.long)

        # Pixels weights based on edges - edge pixels have higher weight
        spatialWeights = torch.from_numpy(spatialWeights).to(self.prec)

        # Distance map for surface loss
        distMap = torch.from_numpy(distMap).to(self.prec)

        # Centers
        pupil_center = torch.from_numpy(pupil_center).to(torch.float32).to(self.prec)
        iris_center = torch.from_numpy(elParam[0][:2]).to(self.prec) if not cond[3] else pupil_center.clone()

        cond = torch.from_numpy(cond).to(self.prec).to(torch.bool)
        imInfo = torch.from_numpy(imInfo).to(torch.long)

        # Generate normalized pupil and iris information
        H = np.array([[2/img.shape[2], 0, -1], [0, 2/img.shape[1], -1], [0, 0, 1]])
        iris_pts, iris_norm = get_ellipse_info(elParam[0], H, cond[3])
        pupil_pts, pupil_norm = get_ellipse_info(elParam[1], H, cond[2])

        elNorm = np.stack([iris_norm, pupil_norm], axis=0) # Respect iris first policy

        elNorm = torch.from_numpy(elNorm).to(self.prec)
        return (img, label, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo)

    def readImage(self, idx):
        '''
        Read an individual image and all its properties using partial loading
        Note: Iris first policy for all data
        '''
        im_num  = self.imList[idx, 0]
        archNum = self.imList[idx, 1]
        archStr = self.arch[archNum]

        path2h5 = os.path.join(self.path2data, str(archStr)+'.h5')
        f = h5py.File(path2h5, 'r')

        # Read information
        I = f['Images'][im_num, ...]
        pupil_center = f['pupil_loc'][im_num, ...] if f['pupil_loc'].__len__() != 0 else -np.ones(2, )
        mask_noSkin = f['Masks_noSkin'][im_num, ...] if f['Masks_noSkin'].__len__() != 0 else -np.ones(I.shape[:2])
        pupil_param = f['Fits']['pupil'][im_num, ...] if f['Fits']['pupil'].__len__() != 0 else -np.ones(5, )
        iris_param = f['Fits']['iris'][im_num, ...] if f['Fits']['iris'].__len__() != 0 else -np.ones(5, )
        f.close()

        # Generate conditions based on available annotations
        cond1 = np.all(pupil_center == -1)
        cond2 = np.all(mask_noSkin == -1) or np.all(mask_noSkin == 0)
        cond3 = np.all(pupil_param == -1)
        cond4 = np.all(iris_param == -1)
        cond = np.array([cond1, cond2, cond3, cond4])

        return I, mask_noSkin, [iris_param, pupil_param], pupil_center, cond, self.imList[idx, :]

def listDatasets(AllDS):
    dataset_list = np.unique(AllDS['dataset'])
    subset_list = np.unique(AllDS['subset'])
    print('Subsets available.')
    return (dataset_list, subset_list)

def readArchives(path2arc_keys):
    D = os.listdir(path2arc_keys)
    AllDS = {'archive': [], 'pupil_loc': [], 'dataset': [], 'im_num': [], 'subset': []}
    for chunk in D:

        # Load archive key
        chunkData = scio.loadmat(os.path.join(path2arc_keys, chunk))
        N = np.size(chunkData['archive'])
        pupil_loc = chunkData['pupil_loc']

        if not chunkData['subset']:
            print('{} does not have subsets.'.format(chunkData['dataset']))
            chunkData['subset'] = 'none'

        if type(pupil_loc) is list:
            # Replace pupil locations with -1
            print('{} does not have pupil center locations. Len: {}'.format(
                    chunkData['dataset'],
                    len(chunkData['pupil_loc'])))
            pupil_loc = -1*np.ones((N, 2))

        loc = np.arange(0, N)
        res = np.flip(chunkData['resolution'], axis=1) # Flip the resolution
        AllDS['im_num'].append(loc)
        AllDS['archive'].append(chunkData['archive'].reshape(-1)[loc])
        AllDS['pupil_loc'].append(pupil_loc[loc, :]/res[loc, :])
        AllDS['dataset'].append(np.repeat(chunkData['dataset'], N))
        AllDS['subset'].append(np.repeat(chunkData['subset'], N))

    # Concat all entries into one giant list
    for key, val in AllDS.items():
        AllDS[key] = np.concatenate(val, axis=0)
    return AllDS

def rmDataset(AllDS, rmSet):
    '''
    Remove datasets.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [True if simple_string(ele)  is simple_string(rmSet) else False for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['dataset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData

def selDataset(AllDS, selSet):
    '''
    Select datasets of interest.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [False if simple_string(ele) in simple_string(selSet) else True for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['dataset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData

def selSubset(AllDS, selSubset):
    '''
    Select subsets of interest.
    '''
    dsData = copy.deepcopy(AllDS)
    subset_list = listDatasets(dsData)[1]
    loc = [False if simple_string(ele) in simple_string(selSubset) else True for ele in subset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['subset'] == subset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData

def rmEntries(AllDS, ent):
    dsData = copy.deepcopy(AllDS)
    dsData['pupil_loc'] = AllDS['pupil_loc'][~ent, :]
    dsData['im_num'] = AllDS['im_num'][~ent, ]
    dsData['archive'] = AllDS['archive'][~ent, ]
    dsData['dataset'] = AllDS['dataset'][~ent, ]
    dsData['subset'] = AllDS['subset'][~ent, ]
    return dsData

def generate_strat_indices(AllDS):
    '''
    Removing images with pupil center values which are 10% near borders.
    Does not remove images with a negative pupil center.
    Returns the indices and a pruned data record.
    '''
    loc_oBounds = (AllDS['pupil_loc'] < 0.10) | (AllDS['pupil_loc'] > 0.90)
    loc_oBounds = np.sum(loc_oBounds, 1).squeeze().astype(np.bool)
    loc_nExist = AllDS['pupil_loc'] < 0
    loc_nExist = np.sum(loc_nExist, 1).squeeze().astype(np.bool)
    loc = loc_oBounds & ~loc_nExist # Location of images to remove
    AllDS = rmEntries(AllDS, loc)

    # Generate 2D histogram of pupil centers
    numBins = 5
    _, edgeList = np.histogramdd(AllDS['pupil_loc'], bins=numBins)
    xEdges, yEdges = edgeList

    archNum = np.unique(AllDS['archive'],
                        return_index=True,
                        return_inverse=True)[2]

    # Bin the pupil center location and return that bin ID
    binx = np.digitize(AllDS['pupil_loc'][:, 0], xEdges, right=True)
    biny = np.digitize(AllDS['pupil_loc'][:, 1], yEdges, right=True)

    # Convert 2D bin locations into indices
    indx = np.ravel_multi_index((binx, biny, archNum),
                                (numBins+1, numBins+1, np.max(archNum)+1))
    indx = indx - np.min(indx)

    # Remove entries which occupy a single element in the grid
    print('Original # of entries: {}'.format(np.size(binx)))
    countInfo = np.unique(indx, return_counts=True)

    for rmInd in np.nditer(countInfo[0][countInfo[1] <= 5]):
        ent = indx == rmInd
        indx = indx[~ent]
        AllDS = copy.deepcopy(rmEntries(AllDS, ent))
    print('# of entries after stratification: {}'.format(np.size(indx)))
    return indx, AllDS

def generate_fileList(AllDS, mode='vanilla', notest=True):
    indx, AllDS = generate_strat_indices(AllDS)

    archNum = np.unique(AllDS['archive'],
                        return_index=True,
                        return_inverse=True)[2]

    feats = np.stack([AllDS['im_num'], archNum, indx], axis=1)
    validPerc = .20

    if 'vanilla' in mode:
        # vanilla splits from the selected datasets.
        # Stratification by pupil center and dataset.
        params = re.findall('\d+', mode)
        if len(params) == 1:
            trainPerc = float(params[0])/100
            print('Training data set to {}%. Validation data set to {}%.'.format(
                        int(100*trainPerc), int(100*validPerc)))
        else:
            trainPerc = 1 - validPerc
            print('Training data set to {}%. Validation data set to {}%.'.format(
                        int(100*trainPerc), int(100*validPerc)))

        data_div = Datasplit(1, AllDS['archive'])

        if not notest:
            # Split into train and test
            train_feats, test_feats = train_test_split(feats,
                                                train_size = trainPerc,
                                                stratify = indx)
        else:
            # Do not split into train and test
            train_feats = feats
            test_feats = []

        # Split training further into validation
        train_feats, valid_feats = train_test_split(train_feats,
                                                    test_size = 0.2,
                                                    stratify = train_feats[:, -1])
        data_div.assignIdx(0, train_feats, valid_feats, test_feats)

    if 'fold' in mode:
        # K fold validation.
        K = int(re.findall('\d+', mode)[0])

        data_div = Datasplit(K, AllDS['archive'])
        skf = StratifiedKFold(n_splits=K, shuffle=True)
        train_feats, test_feats = train_test_split(feats,
                                            train_size = 1 - validPerc,
                                            stratify = indx)
        i=0
        for train_loc, valid_loc in skf.split(train_feats, train_feats[:, -1]):
            data_div.assignIdx(i, train_feats[train_loc, :],
                            train_feats[valid_loc, :],
                            test_feats)
            i+=1

    if 'none' in mode:
        # No splits. All images are placed in train, valid and test.
        # This process ensure's there no confusion.
        data_div = Datasplit(1, AllDS['archive'])
        data_div.assignIdx(0, feats, feats, feats)

    return data_div

def generateIdx(samplesList, batch_size):
    '''
    Takes in 2D array <samplesList>
    samplesList: 1'st dimension image number
    samplesList: 2'nd dimension hf5 file number
    batch_size: Number of images to be present in a batch
    If no entries are found, generateIdx will return an empty list of batches
    '''
    if np.size(samplesList) > 0:
        num_samples = samplesList.shape[0]
        num_batches = np.ceil(num_samples/batch_size).astype(np.int)
        np.random.shuffle(samplesList) # random.shuffle works on the first axis
        batchIdx_list = []
        for i in range(0, num_batches):
            y = (i+1)*batch_size if (i+1)*batch_size<num_samples else num_samples
            batchIdx_list.append(samplesList[i*batch_size:y, :])
    else:
        batchIdx_list = []
    return batchIdx_list

def foldInfo():
    D = {'train_idx': [], 'valid_idx': [], 'test_idx': []}
    return D

class Datasplit():
    def __init__(self, K, archs):
        self.splits = K
        self.folds = [foldInfo() for i in range(0, self.splits)]
        self.arch = np.unique(archs)

    def assignIdx(self, foldNum, train_idx, valid_idx, test_idx):
        # train, valid and test idx contains image number, h5 file and stratify index
        self.checkUnique(train_idx)
        self.checkUnique(valid_idx)
        self.checkUnique(test_idx)

        self.folds[foldNum]['train_idx'] = train_idx[:, :2] if type(train_idx) is not list else []
        self.folds[foldNum]['valid_idx'] = valid_idx[:, :2] if type(valid_idx) is not list else []
        self.folds[foldNum]['test_idx'] = test_idx[:, :2] if type(test_idx) is not list else []

    def checkUnique(self, ID):
        if type(ID) is not list:
            imNums = ID[:, 0]
            chunks = ID[:, 1]
            chunks_present = np.unique(chunks)
            for chunk in chunks_present:
                loc = chunks == chunk
                unq_flg = np.size(np.unique(imNums[loc])) != np.size(imNums[loc])
                if unq_flg:
                    print('Not unique! WARNING')

if __name__=="__main__":
    # This scripts verifies all datasets and returns the total number of images
    # Run sandbox.py to verify dataloader.
    path2data = '/media/rakshit/Monster/Datasets'
    path2arc_keys = os.path.join(path2data, 'MasterKey')

    AllDS = readArchives(path2arc_keys)
    datasets_present, subsets_present = listDatasets(AllDS)

    print('Datasets selected ---------')
    print(datasets_present)
    print('Subsets selected ---------')
    print(subsets_present)

    dataDiv_Obj = generate_fileList(AllDS, mode='vanilla')
    np.save('CurCheck', dataDiv_Obj)
    N = [value.shape[0] for key, value in dataDiv_Obj.folds[0].items() if len(value) > 0]
    print('Total number of images: {}'.format(np.sum(N)))
