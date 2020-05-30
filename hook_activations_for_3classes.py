#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:10:57 2020

@author: aaa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:03:44 2020

@author: aaa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections
import os
import sys
import tqdm
import torch
import pickle
import resource
import numpy as np
import matplotlib.pyplot as plt

from args import parse_args
from modelSummary import model_dict
from pytorchtools import load_from_file
from torch.utils.data import DataLoader
from utils import get_nparams, get_predictions
from helperfunctions import mypause, stackall_Dict
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts,getPoint_metric_norm
from utils import getAng_metric,Logger
import pylab

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*10, rlimit[1]))

#%%
if __name__ == '__main__':

    args = parse_args()
#%%
    args.batchsize=4
    args.model='ritnet_v3'
    args.curObj='OpenEDS'
    device=torch.device("cuda")
    torch.cuda.manual_seed(12)
    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        args.useMultiGPU = False
    torch.backends.cudnn.deterministic=False

    if args.model not in model_dict:
        print("Model not found.")
        print("valid models are: {}".format(list(model_dict.keys())))
        exit(1)

    if args.seg2elactivated:
        path_intermediate='_0_0'#'with_seg2el'
    else:
        path_intermediate='_1_0'#'without_seg2el'       
#    path_intermediate='_1_0'
#    if args.expname=='':
    args.expname='RC_e2e_baseline_'+args.model+'_'+args.curObj+path_intermediate#'_0_0'

    LOGDIR = os.path.join(os.getcwd(), 'ExpData', 'logs_0517',\
                          args.model, args.expname)
#    LOGDIR = os.path.join(os.getcwd(), 'logs', args.model, args.expname)
#    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
#    path2writer = os.path.join(LOGDIR, 'TB.lock')
    path2op = os.path.join(os.getcwd(), 'op', str(args.curObj),args.model)
    path2op_mask = os.path.join(os.getcwd(), 'op', str(args.curObj), args.model,'mask')

#%%
    os.makedirs(LOGDIR, exist_ok=True)
#    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
#    os.makedirs(path2writer, exist_ok=True)
    os.makedirs(path2op, exist_ok=True)
    os.makedirs(path2op_mask, exist_ok=True)
    
    
    model = model_dict[args.model]

    checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')
    netDict = load_from_file([checkpointfile, args.loadfile])
    model.load_state_dict(netDict['state_dict'])
  

    print('Parameters: {}'.format(get_nparams(model)))
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)

    f = open(os.path.join(os.getcwd(),'curObjects', 'baseline/cond_'+str(args.curObj)+'.pkl'), 'rb')

    _, _, testObj = pickle.load(f)
    testObj.path2data = os.path.join(args.path2data, 'Dataset', 'All')
    testObj.augFlag = False
    testObj.imList=np.array([[165,0,0],[197,0,0],[241,0,0],[377,0,0]])

    testloader = DataLoader(testObj,
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)
    #%%
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
    	activations[name].append(out.cpu())
    
    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in model.named_modules():
    	if type(m)==nn.Conv2d:
    		# partial to assign the layer name to each hook
    		m.register_forward_hook(partial(save_activation, name))
    #%%
    for bt, batchdata in enumerate(tqdm.tqdm(testloader)):
        img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = batchdata
        out_tup = model(img.to(device).to(args.prec),
                        labels.to(device).long(),
                        pupil_center.to(device).to(args.prec),
                        elNorm.to(device).to(args.prec),
                        spatialWeights.to(device).to(args.prec),
                        distMap.to(device).to(args.prec),
                        cond.to(device).to(args.prec),
                        imInfo[:, 2].to(device).to(torch.long),
                        0.5)
    
    #%% #run some output
      
    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
    
    # just print out the sizes of the saved activations as a sanity check
    for k,v in activations.items():
    	print (k, v.size())
        
    #%%
    def normalize_image(img):
        return (img-np.min(img))/(np.max(img)-np.min(img))
      
      
    plt.figure(figsize=(9,7),dpi=250)
    for i in range(3):
        plt.subplot(331+3*i)
        if i>0:
            plt.imshow(img[i,0]+20*normalize_image(activations['dec.final.conv2'][i,0].detach().numpy()))    
        else:
            plt.imshow(activations['dec.final.conv2'][i,0].detach().numpy())
        plt.xticks([])
        plt.yticks([])
        plt.subplot(331+3*i+1)
        if i>0:
            plt.imshow(img[i,0]+20*normalize_image(activations['dec.final.conv2'][i,1].detach().numpy()))    
        else:
            plt.imshow(activations['dec.final.conv2'][i,1].detach().numpy())
        plt.xticks([])
        plt.yticks([])
        plt.subplot(331+3*i+2)
        if i>0:
            plt.imshow(img[i,0]+20*normalize_image(activations['dec.final.conv2'][i,2].detach().numpy()))    
        else:
            plt.imshow(activations['dec.final.conv2'][i,2].detach().numpy())
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
#    plt.savefig('../ECCV 2020/activation_ritnetv2_3_class_openeds.png',dpi=250)
   #%% 
    i=3
    j=int(pupil_center[i,1].numpy())
    ax=plt.figure(figsize=(20,7),dpi=100)
    plt.subplot(121)
    plt.plot(activations['dec.final.conv2'][i,0].detach().numpy()[j,:],'r',linewidth=2,label='Background')
    plt.plot(activations['dec.final.conv2'][i,1].detach().numpy()[j,:],color='b',linewidth=2,label='Iris')
    plt.plot(activations['dec.final.conv2'][i,2].detach().numpy()[j,:],'g',linewidth=2,label='Pupil')
    plt.yticks([])
    
    plt.axvline(iris_center[i,0],linestyle='-.',color='b',label='Iris Center')
    plt.axvline(pupil_center[i,0],linestyle='--',color='g',label='Pupil Center')
#    plt.xlabel('Horizontal line scan through Pupil Center')
    plt.title('EpSeg')
    
#    plt.subplot(122)
#    plt.plot(a,'r',linewidth=2,label='Background')
#    plt.plot(b,linewidth=2,color='b',label='Iris')
#    plt.plot(c,'g',linewidth=2,label='Pupil')
#    plt.yticks([])
    
    plt.axvline(iris_center[i,0],linestyle='-.',color='b',label='Iris Center')
    plt.axvline(pupil_center[i,0],linestyle='--',color='g',label='Pupil Center')
#    plt.xlabel('Horizontal line scan through Pupil Center')
    plt.title('ElSeg')

#    j=int(pupil_center[i,0].numpy())
#    plt.subplot(122)
#    plt.plot(activations['dec.final.conv2'][i,0].detach().numpy()[:,j],label='Background')
#    plt.plot(activations['dec.final.conv2'][i,1].detach().numpy()[:,j],label='Iris')
#    plt.plot(activations['dec.final.conv2'][i,2].detach().numpy()[:,j],label='Pupil')
#    
#    plt.axvline(pupil_center[i,1],color='r',label='Pupil Center')
#    plt.axvline(iris_center[i,1],color='g',label='Iris Center')

    ax.text(0.3,-.05, 'Horizontal line scan through Pupil Center', ha='left',fontsize=pylab.rcParams['axes.labelsize'])
#    plt.legend(bbox_to_anchor=(0, 3),ncol=3) 
    plt.tight_layout()
#    plt.savefig('../ECCV 2020/try.png',dpi=250)
