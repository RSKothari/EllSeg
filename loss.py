#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:05:09 2020

@author: rakshit
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.extmath import cartesian

from utils import create_meshgrid, soft_heaviside, _assert_no_grad, cdist, generaliz_mean

def get_seg2ptLoss(op, gtPts, temperature=1):
    # Custom function to find the center of mass to get detected pupil or iris
    # center
    # op: BXHXW - single channel corresponding to pupil or iris predictions
    B, H, W = op.shape
    wtMap = F.softmax(op.view(B, -1)*temperature, dim=1) # [B, HXW]

    XYgrid = create_meshgrid(H, W, normalized_coordinates=True) # 1xHxWx2

    if str(op.device) == 'cpu':
        xloc = XYgrid[0, :, :, 0].reshape(-1)
        yloc = XYgrid[0, :, :, 1].reshape(-1)
    else:
        xloc = XYgrid[0, :, :, 0].reshape(-1).cuda()
        yloc = XYgrid[0, :, :, 1].reshape(-1).cuda()

    xpos = torch.sum(wtMap*xloc, -1, keepdim=True)
    ypos = torch.sum(wtMap*yloc, -1, keepdim=True)
    predPts = torch.stack([xpos, ypos], dim=1).squeeze()

    loss = F.l1_loss(predPts, gtPts, reduction='none')
    return loss, predPts

def get_segLoss(op, target, spatWts, distMap, cond, alpha):
    # Custom function to iteratively go over each sample in a batch and
    # compute loss.
    # cond: Mask exist -> 1, else 0
    B = op.shape[0]
    loss_seg = []
    for i in range(0, B):
        if cond[i] == 1:
            # Valid mask exists
            l_sl = SurfaceLoss(op[i, ...].unsqueeze(0),
                               distMap[i, ...].unsqueeze(0))
            l_cE = wCE(op[i, ...],
                       target[i, ...],
                       spatWts[i, ...])
            l_gD = GDiceLoss(op[i, ...].unsqueeze(0),
                             target[i, ...].unsqueeze(0),
                             F.softmax)
            loss_seg.append(alpha*l_sl + (1-alpha)*l_gD + l_cE)
    if len(loss_seg) > 0:
        return torch.sum(torch.stack(loss_seg))/torch.sum(cond.to(torch.float32))
    else:
        return 0.0

def get_ptLoss(ip_vector, target_vector, cond):
    # Custom function to iteratively find L1 distance over valid samples
    # Note, pupil centers are assumed to be normalized between -1 and 1
    B = ip_vector.shape[0]
    loss_pt = []
    for i in range(0, B):
        if cond[i] == 1:
            # Valid entry
            loss_pt.append(F.l1_loss(ip_vector[i, ...],
                                     target_vector[i, ...]))
    if len(loss_pt) > 0:
        return torch.sum(torch.stack(loss_pt))/torch.sum(cond.to(torch.float32))
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

        # loss = F.kl_div(F.log_softmax(x, dim=1),
        #                 torch.ones(B, C).cuda()/C)

        loss = F.l1_loss(F.softmax(x, dim=1), torch.ones(B, C).cuda()/C)
    else:
        # Else, return the secondary loss
        loss = F.cross_entropy(x, gt)
    return loss

def get_seg2elLoss(opSeg, opEl, loc_seg_ok):
    # Correction loss based on regressed ellipse fit. Higher the overlap,
    # the smaller the loss
    # opSeg: Segmentation output [B, H, W]
    # opEl: Regressed Ellipse output [B, 5]
    # loc_seg_ok: Samples with existing segmentation

    loss = 0
    B, H, W = opSeg.shape
    opSeg = opSeg.to(torch.float32)

    mesh = create_meshgrid(H, W, normalized_coordinates=True).squeeze().cuda() # 1xHxWx2
    mesh.requires_grad = False

    for i in range(0, B):
        if loc_seg_ok[i]:
            X =(mesh[..., 0]-opEl[i, 0])*torch.cos(opEl[i,-1]) +\
                (mesh[..., 1]-opEl[i, 1])*torch.sin(opEl[i,-1])
            Y = -(mesh[..., 0]-opEl[i, 0])*torch.sin(opEl[i,-1]) +\
                (mesh[..., 1]-opEl[i, 1])*torch.cos(opEl[i,-1])
            posmask = ((X/opEl[i, 2])**2 + (Y/opEl[i, 3])**2) - 1
            negmask = -posmask
            posmask = soft_heaviside(posmask, sc=64, mode=3) # Positive outside the ellipse
            negmask = soft_heaviside(negmask, sc=64, mode=3) # Positive inside the ellipse
            loss += (F.binary_cross_entropy(posmask,1-opSeg[i, ...]) + \
                     F.binary_cross_entropy(negmask, opSeg[i, ...]))
    return loss/torch.sum(loc_seg_ok) if torch.sum(loc_seg_ok) else 0.0

def get_selfConsistency(opSeg, opEl, loc_seg_ok):
    # Correction loss based on self consistency KL divergence.
    # opSeg: logSoftmax'ed output channel correspond to ellipse in question
    loss = 0.0

    opSeg = F.log_softmax(opSeg, dim=1)
    B, _, H, W = opSeg.shape
    mesh = create_meshgrid(H, W, normalized_coordinates=True).squeeze().cuda() # 1xHxWx2
    mesh.requires_grad = False

    irisEl = opEl[:, :5]
    pupilEl = opEl[:, 5:]

    for i in range(0, B):
        if loc_seg_ok[i]:
            pupMask = get_mask(mesh, pupilEl[i, :])[1]
            loss+=torch.mean(F.kl_div(opSeg[i, 2, ...], pupMask, reduction='none'))
            bgMask = get_mask(mesh, irisEl[i, :])[0]
            loss+=torch.mean(F.kl_div(opSeg[i, 0, ...], bgMask, reduction='none'))
    return loss/torch.sum(loc_seg_ok) if torch.sum(loc_seg_ok) else 0.0

def get_mask(mesh, opEl):
    # posmask: Positive outside the ellipse
    # negmask: Positive inside the ellipse
    X =(mesh[..., 0]-opEl[0])*torch.cos(opEl[-1]) +\
        (mesh[..., 1]-opEl[1])*torch.sin(opEl[-1])
    Y = -(mesh[..., 0]-opEl[ 0])*torch.sin(opEl[-1]) +\
        (mesh[..., 1]-opEl[ 1])*torch.cos(opEl[-1])
    posmask = (X/opEl[ 2])**2 + (Y/opEl[3])**2 - 1
    negmask = 1 - (X/opEl[ 2])**2 - (Y/opEl[3])**2
    posmask = soft_heaviside(posmask, sc=64, mode=3)
    negmask = soft_heaviside(negmask, sc=64, mode=3)
    return posmask, negmask

class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height, resized_width,
                 p=-9,
                 return_2_terms=False):
        super(WeightedHausdorffDistance, self).__init__()
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super(nn.Module, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                          dtype=torch.float32)
        self.max_dist = np.sqrt(resized_height**2 + resized_width**2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(dtype=torch.float32)

        self.return_2_terms = return_2_terms
        self.p = p

    def forward(self, prob_map, gt, orig_sizes):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size
                           of the original images.
                           B is batch size.
                           The size must be in (height, width) format.
        :param orig_widths: List of the original widths for each image
                            in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """

        _assert_no_grad([gt])

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == gt.shape[0]

        self.all_img_locations = self.all_img_locations.to(prob_map.device)
        self.resized_size = self.resized_size.to(prob_map.device)
        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b, :].unsqueeze(0) # Ensure point is [1, 2]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b/self.resized_size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1)*self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1)*gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1/(n_est_pts+1e-6))*torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated)*self.max_dist + p_replicated*d_matrix
            minn = generaliz_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1, terms_2
        else:
            res = terms_1 + terms_2

        return res
