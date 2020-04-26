#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:05:09 2020

@author: rakshit
"""
import torch
import numpy as np
import torch.nn.functional as F

from utils import create_meshgrid

'''
def get_allLoss(op,
                op_hmaps,
                elOut, # Network outputs
                target, # Segmentation targets
                pupil_center, # Pupil center
                hMaps,
                elPts, # Ellipse points
                elNorm,
                elPhi,
                spatWts,
                distMap,
                cond,
                ID,
                alpha):

    B, C, H, W = op.shape
    pred_c = elOut[:, 5:7]

    # Normalize output heatmap. Iris first policy.
    hmaps_iri = op_hmaps[:, :8, ...]
    hmaps_pup = op_hmaps[:, 8:, ...]
    #hmaps_iri = F.log_softmax(op_hmaps[:, :8, ...].view(B, 8, -1), dim=2)
    #hmaps_pup = F.log_softmax(op_hmaps[:, 8:, ...].view(B, 8, -1), dim=2)
    #hmaps_iri = hmaps_iri.view(B, 8, H, W)
    #hmaps_pup = hmaps_pup.view(B, 8, H, W)

    # Segmentation loss
    l_seg = get_segLoss(op, target, spatWts, distMap, cond, alpha)

    # KL: loss
    #l_map_iri = F.kl_div(hmaps_iri, hMaps[:, 0, ...], reduction='batchmean')
    #l_map_pup = F.kl_div(hmaps_pup, hMaps[:, 1, ...], reduction='batchmean')
    l_map_iri = F.l1_loss(hmaps_iri, hMaps[:, 0, ...])
    l_map_pup = F.l1_loss(hmaps_pup, hMaps[:, 1, ...])
    l_map = l_map_iri + l_map_pup

    # Soft argmax
    temp = spatial_softmax_2d(op_hmaps, torch.tensor(50.0))
    pred_lmrks = spatial_softargmax_2d(temp, normalized_coordinates=True)
    iris_lmrks = pred_lmrks[:, :8, :]
    pupil_lmrks = pred_lmrks[:, 8:, :]

    # Compute landmark based losses
    l_pt = get_ptLoss(pred_c, normPts(pupil_center, target.shape[1:]), cond[:, 0])

    l_lmrks_iri = get_ptLoss(iris_lmrks, elPts[:, 0, ...], cond[:, 1])
    l_lmrks_pup = get_ptLoss(pupil_lmrks, elPts[:, 1, ...], cond[:, 1])
    l_lmrks = l_lmrks_iri + l_lmrks_pup

    # Compute seg losses
    l_seg2pt, pred_c_seg = get_seg2ptLoss(op[:, 2, ...],
                                          normPts(pupil_center,
                                                  target.shape[1:]), 1)

    if alpha > 0.5:
        # Enforce ellipse consistency loss
        iris_center = iris_lmrks.mean(dim=1)
        iris_fit = ElliFit(iris_lmrks, iris_center) # Pupil fit
        pupil_fit = ElliFit(pupil_lmrks, pred_c_seg) # Iris fit
        l_fits = get_ptLoss(iris_fit, elPhi[:, 0, :], cond[:, 1]) + \
                 get_ptLoss(pupil_fit, elPhi[:, 1, :], cond[:, 1])
    else:
        l_fits= 0.0
        iris_fit = -torch.ones(B, 5)
        pupil_fit = -torch.ones(B, 5)

    # Compute ellipse losses - F1 loss for valid samples
    l_ellipse = get_ptLoss(elOut, elNorm.view(-1, 10), cond[:, 1])

    print('l_map: {}. l_lmrks: {}. l_ellipse: {}. l_seg2pt: {}. l_pt: {}. l_seg: {}'.format(
        l_map.item(),
        l_lmrks.item(),
        l_ellipse.item(),
        l_seg2pt.item(),
        l_pt.item(),
        l_seg.item()))

    return (l_map + l_lmrks + l_fits + l_ellipse + l_seg2pt + l_pt + 20*l_seg,
            pred_c_seg,
            torch.stack([hmaps_iri,
                         hmaps_pup], dim=1),
            torch.stack([iris_fit,
                         pupil_fit], dim=1))
'''

def get_seg2ptLoss(op, gtPts, temperature):
    # Custom function to find the pupilary center of mass to detected pupil
    # center
    # op: BXHXW - single channel corresponding to pupil
    B, H, W = op.shape
    wtMap = F.softmax(op.view(B, -1)*temperature, dim=1) # [B, HXW]

    XYgrid = create_meshgrid(H, W, normalized_coordinates=True) # 1xHxWx2

    xloc = XYgrid[0, :, :, 0].reshape(-1).cuda()
    yloc = XYgrid[0, :, :, 1].reshape(-1).cuda()

    xpos = torch.sum(wtMap*xloc, -1, keepdim=True)
    ypos = torch.sum(wtMap*yloc, -1, keepdim=True)
    predPts = torch.stack([xpos, ypos], dim=1).squeeze()

    loss = F.l1_loss(predPts, gtPts, reduction='mean')
    return loss, predPts

def get_segLoss(op, target, spatWts, distMap, cond, alpha):
    # Custom function to iteratively go over each sample in a batch and
    # compute loss.
    B = op.shape[0]
    loss_seg = []
    for i in range(0, B):
        if cond[i] == 0:
            # Valid mask exists
            l_sl = SurfaceLoss(op[i, ...].unsqueeze(0), distMap[i, ...].unsqueeze(0))
            l_cE = wCE(op[i, ...], target[i, ...], spatWts[i, ...])
            l_gD = GDiceLoss(op[i, ...].unsqueeze(0),
                             target[i, ...].unsqueeze(0),
                             F.softmax)
            loss_seg.append(alpha*l_sl + (1-alpha)*l_gD + l_cE)
    if len(loss_seg) > 0:
        return torch.sum(torch.stack(loss_seg))/torch.sum(1-cond)
    else:
        return 0.0

def get_ptLoss(pred_c, pupil_center, cond):
    # Custom function to iteratively find L1 distance over valid samples
    # Note, pupil centers are assumed to be normalized between -1 and 1
    B = pred_c.shape[0]
    loss_pt = []
    for i in range(0, B):
        if cond[i] == 0:
            # Valid pupil center
            loss_pt.append(F.l1_loss(pred_c[i, ...],
                                     pupil_center[i, ...]))
    if len(loss_pt) > 0:
        return torch.sum(torch.stack(loss_pt))/torch.sum(1-cond)
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

def conf_Loss(x, gt, flag):
    '''
    x: Input predicted one-hot encoding for dataset identity
    gt: One-hot encoding of target classes
    flag: Either 1 or 0. Please refer to paper "Turning a Blind Eye: Explicit
    Removal of Biases and Variation from Deep Neural Network Embeddings"
    '''
    if flag:
        B, C = x.shape
        # If true, return the confusion loss
        loss = F.kl_div(F.log_softmax(x, dim=1),
                        torch.ones(B, C).cuda()/C)
    else:
        # Else, return the secondary loss
        loss = F.cross_entropy(x, gt)
    return loss

def selfCorr_seg2el(opSeg, opEl, dims):
    # Self correction loss based on regressed ellipse fit. Higher the overlap,
    # the more negative the loss function will be
    # opSeg: Segmentation output [B, 3, H, W]
    # dims: Segmentation output dims
    # opEl: Regressed Ellipse output [B, 5]

    loss = 0
    B, C, H, W = opSeg.shape
    opSeg = F.softmax(opSeg, dim=1)
    mesh = create_meshgrid(H, W, normalized_coordinates=True).squeeze().cuda() # 1xHxWx2
    mesh.requires_grad = False
   
    for i in range(0, B):
        X = (mesh[..., 0] - opEl[i, 0])*torch.cos(opEl[i, -1]) + (mesh[..., 1] - opEl[i, 1])*torch.sin(opEl[i, -1])
        Y = -(mesh[..., 0] - opEl[i, 0])*torch.sin(opEl[i, -1]) + (mesh[..., 1] - opEl[i, 1])*torch.cos(opEl[i, -1])
        wtMat = ((X/opEl[i, 2])**2 + (Y/opEl[i, 3])**2 - 1)*(2/(H**2+W**2)**0.5) # Negative inside the ellipse
        mask = opSeg[:, dims, ...] if len(dims)==1 else torch.sum(opSeg[:, dims, ...], dim=1)
        loss += torch.sum(wtMat*(2*mask - 1))/(H*W) # 2*posMask -1 creates a SVM like seperation between positive and negative classes
    return loss/B
