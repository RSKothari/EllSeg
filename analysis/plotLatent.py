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

path2data = r'C:\Users\Rudra\Documents\Python Scripts\giw_e2e\op\0'
path2pkl = os.path.join(path2data, 'opDict.pkl')

f = open(path2pkl, 'rb')
opDict = pickle.load(f)
f.close()
pca_do = PCA(n_components='mle', svd_solver='full', whiten=False)
pca_do.fit(opDict['code'])
code = pca_do.transform(opDict['code'])

print('Old shape: {}'.format(opDict['code'].shape))
print('New shape: {}'.format(code.shape))

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

fig, ax1 = plt.subplots()
ax1.scatter(reps[:, 0], reps[:, 1])