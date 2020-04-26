# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:12:30 2020

@author: Rudra
"""

import os
import glob
import torch

path2logs = os.path.join('..', 'logs', 'ritnet')

strSys = 'RC'
cond = [0, 1, 2]
selfCorr = [0, 1]
opDict = {'state_dict':[], 'epoch': 0}
for i in cond:
    for j in selfCorr:
        strModel = '{}_e2e_{}_{}_0'.format(strSys, i, j)
        path2checkpoint = os.path.join(path2logs, strModel, 'checkpoints', 'checkpoint.pt')
        netDict = torch.load(path2checkpoint)
        if 'state_dict' in netDict.keys():
            stateDict = netDict['state_dict']
        else:
            stateDict = netDict
        opDict['state_dict'] = {k: v for k, v in stateDict.items() if 'dsIdentify_lin' not in k}
        strOut = '{}_e2e_{}_{}_1'.format(strSys, i, j)
        path2checkpoint_out = os.path.join(path2logs, strOut, 'checkpoints', 'checkpoint.pt')
        torch.save(opDict, path2checkpoint_out)
        print('Success. {} -> {}'.format(path2checkpoint, path2checkpoint_out))
