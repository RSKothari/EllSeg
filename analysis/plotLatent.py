# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:56:32 2020

@author: Rudra
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

path2data = r'C:\Users\Rudra\Documents\Python Scripts\giw_e2e\op\2'
path2pkl = os.path.join(path2data, 'opDict.pkl')

f = open(path2pkl, 'rb')
opDict = pickle.load(f)
f.close()
pca_do = PCA(n_components=60, svd_solver='full', whiten=False)
pca_do.fit(opDict['code'])
code = pca_do.transform(opDict['code'])

print('Old shape: {}'.format(opDict['code'].shape))
print('New shape: {}'.format(code.shape))

#%%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(0, code.shape[1]),
         pca_do.explained_variance_ratio_, 'g-')
ax2.plot(np.arange(0, code.shape[1]),
         np.cumsum(pca_do.explained_variance_ratio_),'b-')
plt.grid(b=True, which='both', axis='both')
plt.xlabel('Dimensions')
plt.ylabel('Explained covariance')

tsne_do = TSNE(n_components=2, n_jobs=-1)
reps = tsne_do.fit_transform(code)

#%%
dataset = [str(ele).split('_')[0] for ele in np.nditer(opDict['archName'])]
dataset_ID = np.unique(dataset, return_inverse=True)[1]

fig, ax1 = plt.subplots()
sct = ax1.scatter(reps[:, 0], reps[:, 1],
                  c=dataset_ID,
                  alpha=0.3)
ax1.legend(*sct.legend_elements(),
           title='subsets')
ax1.grid(True)

#%%
subsetNames, subsetID = np.unique(opDict['archName'], return_inverse=True)
dsNames = [str(ele).split('_')[0] for ele in np.nditer(subsetNames)]
dsNames = np.unique(dsNames)

print('Datasets present: {}'.format(dsNames))

loc = np.array([True if ele == 'OpenEDS' else False for ele in dataset ])

fig, ax = plt.subplots()
ax.hist(opDict['scores']['seg_dst'][loc], bins=100)
ax.set_xlabel('Pupil center error (pixels)')
ax.set_ylabel('Counts')

print('Median:{}. Mean: {}. STD: {}.'.format(np.median(opDict['scores']['seg_dst'][loc]),
                                             np.mean(opDict['scores']['seg_dst'][loc]),
                                             np.std(opDict['scores']['seg_dst'][loc])))