#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:05:09 2020

@author: rakshit
"""
import torch
import numpy as np
import torch.nn.functional as F

def get_segLoss(op, target, spatWts, distMap, cond, alpha):
    # Custom function to iteratively go over each sample in a batch and
    # compute loss.
    B = op.shape[0]
    loss_seg = []
    for i in range(0, B):
        if cond[i, 1] == 0:
            # Valid mask exists
            l_sl = SurfaceLoss(op[i, ...].unsqueeze(0), distMap[i, ...].unsqueeze(0))
            l_cE = wCE(op[i, ...], target[i, ...], spatWts[i, ...])
            l_gD = GDiceLoss(op[i, ...].unsqueeze(0),
                             target[i, ...].unsqueeze(0),
                             F.softmax)
            loss_seg.append(alpha*l_sl + (1-alpha)*l_gD + l_cE)
    if len(loss_seg) > 0:
        return torch.sum(torch.stack(loss_seg))/torch.sum(1-cond[:, 1])
    else:
        return 0.0
            
def get_ptLoss(pred_c, pupil_center, cond):
    # Custom function to iteratively find L1 distance over valid samples
    # Note, pupil centers are assumed to be normalized between -1 and 1
    B = pred_c.shape[0]
    loss_pt = []
    for i in range(0, B):
        if cond[i, 0] == 0:
            # Valid pupil center
            loss_pt.append(F.l1_loss(pred_c[i, ...],
                                     pupil_center[i, ...]))
    if len(loss_pt) > 0:
        return torch.sum(torch.stack(loss_pt))/torch.sum(1-cond[:, 0])
    else:
        return 0.0

def SurfaceLoss(x, distmap):
    # For classes with no groundtruth, distmap would ideally be filled with 0s
    x = torch.softmax(x, dim=1)
    score = x.flatten(start_dim=2)*distmap.flatten(start_dim=2)
    score = torch.mean(score, dim=2) # Mean between pixels per channel
    score = torch.mean(score, dim=1) # Mean between channels
    return score

def GDiceLoss(ip, target, norm=F.softmax):
    
    mxLabel = ip.shape[1]
    allClasses = np.arange(mxLabel, )
    labelsPresent = np.unique(target.cpu().numpy())
    
    Label = (np.arange(mxLabel) == target.cpu().numpy()[..., None]).astype(np.uint8)
    Label = np.moveaxis(Label, 3, 1)
    target = torch.from_numpy(Label).cuda().to(ip.dtype)
    
    loc_rm = np.where(~np.in1d(allClasses, labelsPresent))[0]
    
    assert ip.shape == target.shape
    ip = norm(ip, dim=1) # Softmax or Sigmoid over channels
    ip = torch.flatten(ip, start_dim=2, end_dim=-1)
    target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(ip.dtype)
    numerator = ip*target
    denominator = ip + target

    # For classes which do not exist in target but exist in input, set weight=0
    class_weights = 1./(torch.sum(target, dim=2)**2).clamp(1e-5)
    if loc_rm.size > 0:
        for i in np.nditer(loc_rm):
            class_weights[:, i] = 0
    A = class_weights*torch.sum(numerator, dim=2)
    B = class_weights*torch.sum(denominator, dim=2)
    dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
    return torch.mean(1 - dice_metric.clamp(1e-5))

def wCE(ip, target, spatWts):
    mxLabel = ip.shape[0]
    allClasses = np.arange(mxLabel, )
    labelsPresent = np.unique(target.cpu().numpy())
    rmIdx = allClasses[~np.in1d(allClasses, labelsPresent)]
    if rmIdx.size > 0:
        loss = spatWts.view(1, -1)*F.cross_entropy(ip.view(1, mxLabel, -1),
                                                    target.view(1, -1),
                                                    ignore_index=rmIdx.item())
    else:
        loss = spatWts.view(1, -1)*F.cross_entropy(ip.view(1, mxLabel, -1),
                                                    target.view(1, -1))
    loss = torch.mean(loss)
    return loss
    
    