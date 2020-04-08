#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:17:32 2020

@author: rakshit
"""

# This file contains definitions which are not applicable in regular scenarios.
# For general purposes functions, classes and operations - please use helperfunctions.
import os
import tqdm
import copy
import torch
import numpy as np
import torch.nn.functional as F

from torchvision.utils import make_grid
from skimage.draw import circle
from typing import Optional
from sklearn import metrics

def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_predictions(output):
    '''
    Parameters
    ----------
    output : torch.tensor
        [B, C, *] tensor. Returns the argmax for one-hot encodings.

    Returns
    -------
    indices : torch.tensor
        [B, *] tensor.

    '''
    bs,c,h,w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs,h,w) # bs x h x w
    return indices

class Logger():
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.dirname = dirname
        self.log_file = open(output_name, 'a+')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write_silent(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print (msg)
    def write_summary(self,msg):
        self.log_file.write(msg)
        self.log_file.write('\n')
        self.log_file.flush()
        print (msg)

def getSeg_metrics(y_true, y_pred, cond):
    '''
    Iterate over each batch and identify which classes are present. If no
    class is present, i.e. all 0, then ignore that score from the average.
    Note: This function computes the nan mean. This is because datasets may not
    have all classes present.
    '''
    assert y_pred.ndim==3, 'Incorrect number of dimensions'
    assert y_true.ndim==3, 'Incorrect number of dimensions'

    cond = cond.astype(np.bool)
    B = y_true.shape[0]
    score_list = []
    for i in range(0, B):
        labels_present = np.unique(y_true[i, ...])
        score_vals = np.empty((3, ))
        score_vals[:] = np.nan
        if not cond[i, 1]:
            score = metrics.jaccard_score(y_true[i, ...].reshape(-1),
                                          y_pred[i, ...].reshape(-1),
                                          labels=labels_present,
                                          average=None)
            # Assign score to relevant location
            for j, val in np.ndenumerate(labels_present):
                score_vals[val] = score[j]
        score_list.append(score_vals)
    score_list = np.stack(score_list, axis=0)
    score_list_clean = score_list[~cond[:, 1], :] # Only select valid entries
    perClassIOU = np.nanmean(score_list_clean, axis=0) if len(score_list_clean) > 0 else np.nan*np.ones(3, )
    meanIOU = np.nanmean(perClassIOU) if len(score_list_clean) > 0 else np.nan
    return meanIOU, perClassIOU, score_list

def getPoint_metric(y_true, y_pred, cond, sz, do_unnorm):
    # Unnormalize predicted points
    if do_unnorm:
        y_pred = unnormPts(y_pred, sz)

    cond = cond.astype(np.bool)
    flag = (~cond[:, 0]).astype(np.float)
    dist = metrics.pairwise_distances(y_true, y_pred, metric='euclidean')
    dist = flag*np.diag(dist)
    return (np.sum(dist)/np.sum(flag) if np.any(flag) else np.nan,
            dist)

def conf_Loss(x, gt, flag):
    '''
    x: Input predicted one-hot encoding for dataset identity
    gt: One-hot encoding of target classes
    flag: Either 1 or 0. Please refer to paper "Turning a Blind Eye: Explicit
    Removal of Biases and Variation from Deep Neural Network Embeddings"
    '''
    if flag:
        # If true, return the confusion loss
        loss = torch.mean(torch.mean(-F.log_softmax(x, dim=1), dim=1))
    else:
        # Else, return the secondary loss
        loss = F.cross_entropy(x, gt)
    return loss

def generateImageGrid(I, mask, pupil_center, cond, override=False):
    '''
    Parameters
    ----------
    I : numpy array [B, H, W]
        A batchfirst array which holds images.
    mask : numpy array [B, H, W]
        A batch first array which holds for individual pixels.
    pupil_center : numpy array [B, 2]
        Identified pupil center for plotting.
    cond : numpy array [B, 5]
        A flag array which holds information about what information is present.
    override : bool, optional
        An override flag which plots data despite being demarked in the flag
        array. Generally used during testing.
        The default is False.

    Returns
    -------
    I_o : numpy array [Ho, Wo]
        Returns an array holding concatenated images from the input overlayed
        with segmentation mask, pupil center and pupil ellipse.
    '''
    I_o = []
    for i in range(0, min(16, cond.shape[0])):
        im = I[i, ...].squeeze()
        im = np.stack([im for i in range(0, 3)], axis=2)

        if (not cond[i, 1]) or override:
            # If masks exists
            rr, cc = np.where(mask[i, ...] == 1)
            im[rr, cc, ...] = np.array([-1, 1, -1]) # Red
            rr, cc = np.where(mask[i, ...] == 2)
            im[rr, cc, ...] = np.array([1, 1, -1])

        if (not cond[i, 0]) or override:
            # If pupil center exists
            rr, cc = circle(pupil_center[i, 1].clip(6, im.shape[0]-6),
                            pupil_center[i, 0].clip(6, im.shape[1]-6),
                            5)
            im[rr, cc, ...] = 1
        I_o.append(im)
    I_o = np.stack(I_o, axis=0)
    I_o = np.moveaxis(I_o, 3, 1)
    I_o = make_grid(torch.from_numpy(I_o), nrow=4)
    I_o = I_o - I_o.min()
    I_o = I_o/I_o.max()
    return I_o

def normPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    pts_o[:, 0] = 2*(pts_o[:, 0]/sz[1]) - 1
    pts_o[:, 1] = 2*(pts_o[:, 1]/sz[0]) - 1
    return pts_o

def unnormPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    pts_o[:, 0] = 0.5*sz[1]*(pts_o[:, 0] + 1)
    pts_o[:, 1] = 0.5*sz[0]*(pts_o[:, 1] + 1)
    return pts_o

def lossandaccuracy(args, loader, model, alpha, device):
    '''
    A function to compute validation loss and performance

    Parameters
    ----------
    loader : torch loader
        Custom designed loader found in the helper functions.
    model : torch net
        Initialized model which needs to be validated againt loader.
    alpha : Learning rate factor. Refer to RITNet paper for more information.
        constant.

    Returns
    -------
    TYPE
        validation score.

    '''
    epoch_loss = []
    ious = []
    dists = []
    dists_seg = []
    model.eval()
    latent_codes = []
    with torch.no_grad():
        for bt, batchdata in enumerate(tqdm.tqdm(loader)):
            img, labels, spatialWeights, distMap, pupil_center, cond, imInfo = batchdata
            output, latent, pred_center, seg_center, loss = model(img.to(device).to(args.prec),
                                                              labels.to(device).long(),
                                                              pupil_center.to(device).to(args.prec),
                                                              spatialWeights.to(device).to(args.prec),
                                                              distMap.to(device).to(args.prec),
                                                              cond.to(device).to(args.prec),
                                                              imInfo[:, 2].to(device).to(torch.long),
                                                              alpha)
            latent_codes.append(latent.detach().cpu())
            loss = loss.mean() if args.useMultiGPU else loss
            epoch_loss.append(loss.item())

            ptDist = getPoint_metric(pupil_center.numpy(),
                                     pred_center.detach().cpu().numpy(),
                                     cond.numpy(),
                                     img.shape[2:],
                                     True)[0] # Unnormalizes the points

            ptDist_seg = getPoint_metric(pupil_center.numpy(),
                                         seg_center.detach().cpu().numpy(),
                                         cond.numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points
            dists.append(ptDist)
            dists_seg.append(ptDist_seg)

            predict = get_predictions(output)
            iou = getSeg_metrics(labels.numpy(),
                                 predict.numpy(),
                                 cond.numpy())[1]
            ious.append(iou)
    ious = np.stack(ious, axis=0)
    return np.mean(epoch_loss), np.nanmean(ious, 0), dists, dists_seg, latent_codes
