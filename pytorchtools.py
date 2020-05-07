#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:04:18 2019
"""
##https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import os
import sys
import torch
import numpy as np

# Useful PyTorch classes
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # Modified by Rakshit Kothari
    def __init__(self,
                patience=7,
                verbose=False,
                delta=0,
                mode='min',
                fName = 'checkpoint.pt',
                path2save = '/home/rakshit/Documents/Python_Scripts/RIT-Eyes/RK/checkpoints'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            fName (str): Name of the checkpoint file.
            path2save (str): Location of the checkpoint file.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta
        self.path2save = path2save
        self.fName = fName
        self.mode = mode

    def __call__(self, val_loss, model):
        score = -val_loss if self.mode =='min' else val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < (self.best_score + self.delta):
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_dict):
        '''Saves model when validation loss decreases.'''
        if self.verbose and self.mode == 'min':
            print('Validation metric decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        elif self.verbose and self.mode == 'max':
            print('Validation metric increased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model_dict, os.path.join(self.path2save, self.fName))
        self.val_loss_min = val_loss

# Useful PyTorch functions
def weights_init(ObjVar):
    # Function to initialize weights
    for name, val in ObjVar.named_parameters():
        if 'weight' in name and len(val.shape) >= 2:
            torch.nn.init.xavier_normal_(val, gain=1)
        elif 'bias' in name:
            torch.nn.init.zeros_(val)
        elif ('nalu' in name) or ('nac' in name):
            torch.nn.init.zeros_(val)
        elif '_' in name:
            print('{}. Ignoring.'.format(name))
        else:
            print('{}. No init.'.format(name))
    return ObjVar

def partial_weight_loading(net, net_odict):
    # Load all weights which have a matching string.
    # WARNING: This code can break in multiple ways.
    # Use with caution. If you the data loading does
    # not look right, retrain from scratch.
    available_keys = [key for key in net_odict.keys()]
    for name, param in net.named_parameters():
        matchedkey = [key for key in available_keys if name in key]
        if len(matchedkey) == 1:
            if net_odict[matchedkey[0]].data.shape == param.data.shape:
                param.data = net_odict[matchedkey[0]].cpu().data
            else:
                print('Shapes did not match. Ignoring weight: {}.'.format(name))
        else:
            print('Could not match: {}. Ignoring this parameter.'.format(name))
    print('Values loaded!')
    return net

def move_to_multi(model_dict):
    '''
    Convert dictionary of weights and keys
    to a multiGPU format. It simply appends
    a 'module.' in front of keys.
    '''
    multiGPU_dict = {}
    for key, value in model_dict.items():
        multiGPU_dict['module.'+key] = value
    return multiGPU_dict

def move_to_single(model_dict):
    '''
    Convert dictionary of weights and keys
    to a singleGPU format. It removes the
    'module.' in front of keys.
    '''
    singleGPU_dict = {}
    for key, value in model_dict.items():
        singleGPU_dict[key.replace('module.', '')] = value
    return singleGPU_dict

def my_collate(batch):
    '''
    batch: list of information acquired from __getitem__
    '''
    I = torch.stack([item[0] for item in batch], dim=0)
    M = torch.stack([item[1] for item in batch], dim=0)
    M_nS = torch.stack([item[2] for item in batch], dim=0)
    spatW = torch.stack([item[3] for item in batch], dim=0)
    distM = torch.stack([item[4] for item in batch], dim=0)
    subjectID = [item[5] for item in batch]
    fName = [item[6] for item in batch]
    pupilPhi = torch.stack([item[7][0] for item in batch], dim=0)
    irisPhi = torch.stack([item[7][1] for item in batch], dim=0)
    return I, M, M_nS, spatW, distM, subjectID, fName, (pupilPhi, irisPhi)

def load_from_file(paths_file):
    # Loads model weights from paths_file, a tuple of filepaths
    # Sequentially moves from first file, attempts to load and if unsuccessful
    # loads the next file and so on ...
    for path in paths_file:
        if path:
            try:
                netDict = torch.load(path)
                print('File loaded from: {}'.format(path))
                break
            except:
                print('WARNING. Path found but failed to load: {}'.format(path))
        else:
            netDict = {}
    return netDict
