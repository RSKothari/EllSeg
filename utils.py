#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:17:32 2020

@author: rakshit
"""

# This file contains definitions which are not applicable in regular scenarios.
# For general purposes functions, classes and operations - please use helperfunctions.
import os
import cv2
import tqdm
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from skimage import draw
from typing import Optional
from sklearn import metrics
from helperfunctions import my_ellipse

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
        if not cond[i]:
            score = metrics.jaccard_score(y_true[i, ...].reshape(-1),
                                          y_pred[i, ...].reshape(-1),
                                          labels=labels_present,
                                          average=None)
            # Assign score to relevant location
            for j, val in np.ndenumerate(labels_present):
                score_vals[val] = score[j]
        score_list.append(score_vals)
    score_list = np.stack(score_list, axis=0)
    score_list_clean = score_list[~cond, :] # Only select valid entries
    perClassIOU = np.nanmean(score_list_clean, axis=0) if len(score_list_clean) > 0 else np.nan*np.ones(3, )
    meanIOU = np.nanmean(perClassIOU) if len(score_list_clean) > 0 else np.nan
    return meanIOU, perClassIOU, score_list

def getPoint_metric(y_true, y_pred, cond, sz, do_unnorm):
    # Unnormalize predicted points
    if do_unnorm:
        y_pred = unnormPts(y_pred, sz)

    cond = cond.astype(np.bool)
    flag = (~cond).astype(np.float)
    dist = metrics.pairwise_distances(y_true, y_pred, metric='euclidean')
    dist = flag*np.diag(dist)
    return (np.sum(dist)/np.sum(flag) if np.any(flag) else np.nan,
            dist)

def getAng_metric(y_true, y_pred, cond):
    # Assumes the incoming angular measurements are in radians
    cond = cond.astype(np.bool)
    flag = (~cond).astype(np.float)
    dist = np.rad2deg(flag*np.abs(y_true - y_pred))
    return (np.sum(dist)/np.sum(flag) if np.any(flag) else np.nan,
            dist)

def generateImageGrid(I,
                      mask,
                      elNorm,
                      pupil_center,
                      cond,
                      heatmaps=False,
                      override=False):
    '''
    Parameters
    ----------
    I : numpy array [B, H, W]
        A batchfirst array which holds images
    mask : numpy array [B, H, W]
        A batch first array which holds for individual pixels.
    hMaps: numpy array [B, C, N, H, W]
        N is the # of points, C is the category the points belong to (iris or
        pupil). Heatmaps are gaussians centered around point of interest
    elNorm:numpy array [B, C, 5]
        Normalized ellipse parameters
    pupil_center : numpy array [B, 2]
        Identified pupil center for plotting.
    cond : numpy array [B, 5]
        A flag array which holds information about what information is present.
    heatmaps : bool, optional
        Unless specificed, does not show the heatmaps of predicted points
    override : bool, optional
        An override flag which plots data despite being demarked in the flag
        array. Generally used during testing.
        The default is False.

    Returns
    -------
    I_o : numpy array [Ho, Wo]
        Returns an array holding concatenated images from the input overlayed
        with segmentation mask, pupil center and pupil ellipse.

    Note: If masks exist, then ellipse parameters would exist aswell.
    '''
    B, H, W = I.shape
    mesh = create_meshgrid(H, W, normalized_coordinates=True) # 1xHxWx2
    H = np.array([[W/2, 0, W/2], [0, H/2, H/2], [0, 0, 1]])
    I_o = []
    for i in range(0, min(16, cond.shape[0])):
        im = I[i, ...].squeeze() - I[i, ...].min()
        im = cv2.equalizeHist(np.uint8(255*im/im.max()))
        im = np.stack([im for i in range(0, 3)], axis=2)

        if (not cond[i, 1]) or override:
            # If masks exists

            rr, cc = np.where(mask[i, ...] == 1)
            im[rr, cc, ...] = np.array([0, 255, 0]) # Green
            rr, cc = np.where(mask[i, ...] == 2)
            im[rr, cc, ...] = np.array([255, 255, 0]) # Yellow


            # Just for experiments. Please ignore.
            el_iris = elNorm[i, 0, ...]
            X = (mesh[..., 0].squeeze() - el_iris[0])*np.cos(el_iris[-1])+\
                (mesh[..., 1].squeeze() - el_iris[1])*np.sin(el_iris[-1])
            Y = -(mesh[..., 0].squeeze() - el_iris[0])*np.sin(el_iris[-1])+\
                 (mesh[..., 1].squeeze() - el_iris[1])*np.cos(el_iris[-1])
            wtMat = (X/el_iris[2])**2 + (Y/el_iris[3])**2 - 1
            # [rr_i, cc_i] = np.where(wtMat< 0)

            try:
                el_iris = my_ellipse(elNorm[i, 0, ...]).transform(H)[0]
                el_pupil = my_ellipse(elNorm[i, 1, ...]).transform(H)[0]
            except:
                print('Warning: inappropriate ellipses. Defaulting to not break runtime..')
                el_iris = np.array([W/2, H/2, W/8, H/8, 0.0]).astype(np.float32)
                el_pupil = np.array([W/2, H/2, W/4, H/4, 0.0]).astype(np.float32)

            [rr_i, cc_i] = draw.ellipse_perimeter(int(el_iris[1]),
                                              int(el_iris[0]),
                                              int(el_iris[3]),
                                              int(el_iris[2]),
                                              orientation=el_iris[4])
            [rr_p, cc_p] = draw.ellipse_perimeter(int(el_pupil[1]),
                                             int(el_pupil[0]),
                                             int(el_pupil[3]),
                                             int(el_pupil[2]),
                                             orientation=el_pupil[4])
            rr_i = rr_i.clip(6, im.shape[0]-6)
            rr_p = rr_p.clip(6, im.shape[0]-6)
            cc_i = cc_i.clip(6, im.shape[1]-6)
            cc_p = cc_p.clip(6, im.shape[1]-6)

            im[rr_i, cc_i, ...] = np.array([0, 0, 255])
            im[rr_p, cc_p, ...] = np.array([255, 0, 0])

        if (not cond[i, 0]) or override:
            # If pupil center exists
            rr, cc = draw.disk((pupil_center[i, 1].clamp(6, im.shape[0]-6),
                                pupil_center[i, 0].clamp(6, im.shape[1]-6)),
                                 5)
            im[rr, cc, ...] = 255
        I_o.append(im)
    I_o = np.stack(I_o, axis=0)
    I_o = np.moveaxis(I_o, 3, 1)
    I_o = make_grid(torch.from_numpy(I_o).to(torch.float), nrow=4)
    I_o = I_o - I_o.min()
    I_o = I_o/I_o.max()
    return I_o

def normPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    res = pts_o.shape
    pts_o = pts_o.reshape(-1, 2)
    pts_o[:, 0] = 2*(pts_o[:, 0]/sz[1]) - 1
    pts_o[:, 1] = 2*(pts_o[:, 1]/sz[0]) - 1
    pts_o = pts_o.reshape(res)
    return pts_o

def unnormPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    res = pts_o.shape
    pts_o = pts_o.reshape(-1, 2)
    pts_o[:, 0] = 0.5*sz[1]*(pts_o[:, 0] + 1)
    pts_o[:, 1] = 0.5*sz[0]*(pts_o[:, 1] + 1)
    pts_o = pts_o.reshape(res)
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

    scoreType = {'c_dist':[], 'ang_dist': [], 'sc_rat': []}
    scoreTrack = {'pupil': copy.deepcopy(scoreType),
                  'iris': copy.deepcopy(scoreType)}

    model.eval()
    latent_codes = []
    with torch.no_grad():
        for bt, batchdata in enumerate(tqdm.tqdm(loader)):
            img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = batchdata
            op_tup = model(img.to(device).to(args.prec),
                            labels.to(device).long(),
                            pupil_center.to(device).to(args.prec),
                            elNorm.to(device).to(args.prec),
                            spatialWeights.to(device).to(args.prec),
                            distMap.to(device).to(args.prec),
                            cond.to(device).to(args.prec),
                            imInfo[:, 2].to(device).to(torch.long), # Send DS #
                            alpha)

            output, elOut, latent, loss = op_tup
            latent_codes.append(latent.detach().cpu())
            loss = loss.mean() if args.useMultiGPU else loss
            epoch_loss.append(loss.item())

            pred_c_iri = elOut[:, 0:2].detach().cpu().numpy()
            pred_c_pup = elOut[:, 5:7].detach().cpu().numpy()

            # Center distance
            ptDist_iri = getPoint_metric(iris_center.numpy(),
                                         pred_c_iri,
                                         cond[:,0].numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points
            ptDist_pup = getPoint_metric(pupil_center.numpy(),
                                         pred_c_pup,
                                         cond[:,0].numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points

            # Angular distance
            angDist_iri = getAng_metric(elNorm[:, 0, 4].numpy(),
                                        elOut[:,  4].detach().cpu().numpy(),
                                        cond[:, 1].numpy())[0]
            angDist_pup = getAng_metric(elNorm[:, 1, 4].numpy(),
                                        elOut[:, 9].detach().cpu().numpy(),
                                        cond[:, 1].numpy())[0]

            # Scale metric
            gt_ab = elNorm[:, 0, 2:4]
            pred_ab = elOut[:, 2:4].cpu().detach()
            scale_iri = torch.sqrt(torch.sum(gt_ab**2, dim=1)/torch.sum(pred_ab**2, dim=1))
            scale_iri = torch.sum(scale_iri*(~cond[:,1]).to(torch.float32)).item()
            gt_ab = elNorm[:, 1, 2:4]
            pred_ab = elOut[:, 7:9].cpu().detach()
            scale_pup = torch.sqrt(torch.sum(gt_ab**2, dim=1)/torch.sum(pred_ab**2, dim=1))
            scale_pup = torch.sum(scale_pup*(~cond[:,1]).to(torch.float32)).item()

            predict = get_predictions(output)
            iou = getSeg_metrics(labels.numpy(),
                                 predict.numpy(),
                                 cond[:, 1].numpy())[1]
            ious.append(iou)

            # Append to score dictionary
            scoreTrack['iris']['c_dist'].append(ptDist_iri)
            scoreTrack['iris']['ang_dist'].append(angDist_iri)
            scoreTrack['iris']['sc_rat'].append(scale_iri)
            scoreTrack['pupil']['c_dist'].append(ptDist_pup)
            scoreTrack['pupil']['ang_dist'].append(angDist_pup)
            scoreTrack['pupil']['sc_rat'].append(scale_pup)

            ious.append(iou)
    ious = np.stack(ious, axis=0)

    return (np.mean(epoch_loss),
            np.nanmean(ious, 0),
            scoreTrack,
            latent_codes)

def points_to_heatmap(pts, std, res):
    # Given image resolution and variance, generate synthetic Gaussians around
    # points of interest for heat map regression.
    # pts: [B, C, N, 2] Normalized points
    # H: [B, C, N, H, W] Output heatmap
    B, C, N, _ = pts.shape
    pts = unnormPts(pts, res) #
    grid = create_meshgrid(res[0], res[1], normalized_coordinates=False)
    grid = grid.squeeze()
    X = grid[..., 0]
    Y = grid[..., 1]

    X = torch.stack(B*C*N*[X], axis=0).reshape(B, C, N, res[0], res[1])
    X = X - torch.stack(np.prod(res)*[pts[..., 0]], axis=3).reshape(B, C, N, res[0], res[1])

    Y = torch.stack(B*C*N*[Y], axis=0).reshape(B, C, N, res[0], res[1])
    Y = Y - torch.stack(np.prod(res)*[pts[..., 1]], axis=3).reshape(B, C, N, res[0], res[1])

    H = torch.exp(-(X**2 + Y**2)/(2*std**2))
    #H = H/(2*np.pi*std**2) # This makes the summation == 1 per image in a batch
    return H

def ElliFit(coords, mns):
    '''
    Parameters
    ----------
    coords : torch float32 [B, N, 2]
        Predicted points on ellipse periphery
    mns : torch float32 [B, 2]
        Predicted mean of the center points

    Returns
    -------
    PhiOp: The Phi scores associated with ellipse fitting. For more info,
    please refer to ElliFit paper.
    '''
    B = coords.shape[0]

    PhiList = []

    for bt in range(B):
        coords_norm = coords[bt, ...] - mns[bt, ...] # coords_norm: [N, 2]
        N = coords_norm.shape[0]

        x = coords_norm[:, 0]
        y = coords_norm[:, 1]

        X = torch.stack([-x**2, -x*y, x, y, -torch.ones(N, ).cuda()], dim=1)
        Y = y**2

        a = torch.inverse(X.T.matmul(X))
        b = X.T.matmul(Y)
        Phi = a.matmul(b)
        PhiList.append(Phi)
    Phi = torch.stack(PhiList, dim=0)
    return Phi

def spatial_softmax_2d(input: torch.Tensor, temperature: torch.Tensor = torch.tensor(1.0)) -> torch.Tensor:
    r"""Applies the Softmax function over features in each image channel.
    Note that this function behaves differently to `torch.nn.Softmax2d`, which
    instead applies Softmax over features at each spatial location.
    Returns a 2D probability distribution per image channel.
    Arguments:
        input (torch.Tensor): the input tensor.
        temperature (torch.Tensor): factor to apply to input, adjusting the
          "smoothness" of the output distribution. Default is 1.
    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, H, W)`
    """

    batch_size, channels, height, width = input.shape
    x: torch.Tensor = input.view(batch_size, channels, -1)

    x_soft: torch.Tensor = F.softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)

def spatial_softargmax_2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""Computes the 2D soft-argmax of a given input heatmap.
    The input heatmap is assumed to represent a valid spatial probability
    distribution, which can be achieved using
    :class:`~kornia.contrib.dsnt.spatial_softmax_2d`.
    Returns the index of the maximum 2D coordinates of the given heatmap.
    The output order of the coordinates is (x, y).
    Arguments:
        input (torch.Tensor): the input tensor.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`
    Examples:
        >>> heatmaps = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.]]]])
        >>> coords = spatial_softargmax_2d(heatmaps, False)
        tensor([[[1.0000, 2.0000]]])
    """

    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates)
    grid = grid.to(device=input.device, dtype=input.dtype)

    pos_x: torch.Tensor = grid[..., 0].reshape(-1)
    pos_y: torch.Tensor = grid[..., 1].reshape(-1)

    input_flat: torch.Tensor = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y: torch.Tensor = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x: torch.Tensor = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output: torch.Tensor = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2

def soft_heaviside(x, sc, mode):
    '''
    Given an input and a scaling factor (default 64), the soft heaviside
    function approximates the behavior of a 0 or 1 operation in a differentiable
    manner. Note the max values in the heaviside function are scaled to 0.9.
    This scaling is for convenience and stability with bCE loss.
    '''
    sc = torch.tensor([sc]).to(torch.float32).to(x.device)
    if mode==1:
        # Original soft-heaviside
        # Try sc = 64
        return 0.9/(1 + torch.exp(-sc/x))
    elif mode==2:
        # Some funky shit but has a nice gradient
        # Try sc = 0.001
        return 0.45*(1 + (2/np.pi)*torch.atan2(x, sc))
    elif mode==3:
        # Good ol' scaled sigmoid. FUTURE: make sc free parameter
        # Try sc = 8
        return torch.sigmoid(sc*x)
    else:
        print('Mode undefined')

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    return res

class linStack(torch.nn.Module):
    """A stack of linear layers followed by batch norm and hardTanh

    Attributes:
        num_layers: the number of linear layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, bias, actBool, dp):
        super().__init__()

        layers_lin = []
        for i in range(num_layers):
            m = torch.nn.Linear(hidden_dim if i > 0 else in_dim,
                hidden_dim if i < num_layers - 1 else out_dim, bias=bias)
            layers_lin.append(m)
        self.layersLin = torch.nn.ModuleList(layers_lin)
        self.act_func = torch.nn.SELU()
        self.actBool = actBool
        self.dp = torch.nn.Dropout(p=dp)

    def forward(self, x):
        # Input shape (batch, features, *)
        for i, _ in enumerate(self.layersLin):
            x = self.act_func(x) if self.actBool else x
            x = self.layersLin[i](x)
            x = self.dp(x)
        return x

class regressionModule(torch.nn.Module):
    def __init__(self, sizes):
        super(regressionModule, self).__init__()
        inChannels = sizes['enc']['op'][-1]
        self.max_pool = nn.AvgPool2d(kernel_size=2)

        self.c1 = nn.Conv2d(in_channels=inChannels,
                            out_channels=128,
                            bias=True,
                            kernel_size=(2,3))

        self.c2 = nn.Conv2d(in_channels=128,
                            out_channels=128,
                            bias=True,
                            kernel_size=3)

        self.c3 = nn.Conv2d(in_channels=128,
                            out_channels=32,
                            kernel_size=3,
                            bias=False)

        self.l1 = nn.Linear(32*3*5, 256, bias=True)
        self.l2 = nn.Linear(256, 10, bias=True)

        self.c_actfunc = torch.tanh # Center has to be between -1 and 1
        self.param_actfunc = torch.sigmoid # Parameters can't be negative and capped to 1

    def forward(self, x, alpha):
        B = x.shape[0]
        # x: [B, 192, H/16, W/16]
        x = F.leaky_relu(self.c1(x)) # [B, 256, 14, 18]
        x = self.max_pool(x) # [B, 256, 7, 9]
        x = F.leaky_relu(self.c2(x)) # [B, 256, 5, 7]
        x = F.leaky_relu(self.c3(x)) # [B, 32, 3, 5]
        x = x.reshape(B, -1)
        x = self.l2(torch.selu(self.l1(x)))

        pup_c = self.c_actfunc(x[:, 0:2])
        pup_param = self.param_actfunc(x[:, 2:4])
        pup_angle = x[:, 4]
        iri_c = self.c_actfunc(x[:, 5:7])
        iri_param = self.param_actfunc(x[:, 7:9])
        iri_angle = x[:, 9]


        op = torch.cat([pup_c,
                        pup_param,
                        pup_angle.unsqueeze(1),
                        iri_c,
                        iri_param,
                        iri_angle.unsqueeze(1)], dim=1)
        return op

class convBlock(nn.Module):
    def __init__(self, in_c, inter_c, out_c, actfunc):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inter_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = torch.nn.BatchNorm2d(num_features=out_c)
    def forward(self, x):
        x = self.actfunc(self.conv1(x))
        x = self.actfunc(self.conv2(x)) # Remove x if not working properly
        x = self.bn(x)
        return x
'''
class refineModule(nn.Module):
    def __init__(self, N):
        super(refineModule, self).__init__()
        self.c1 =
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, elPred, skips):
        # skips: [smallest to largest]
        el_iris = elPred[:, :5, ...]
        el_pupil = elPred[:, 5:, ...]

        featMap = []
        for sc, skip in enumerate(skips):
            B, C, H, W = skips[sc].shape
            mesh = create_meshgrid(height=H,
                                   width=W,
                                   normalized_coordinates=True)
            mesh.requires_grad = False
            mesh = mesh.to(skip.device)
            mesh = torch.cat([mesh for i in range(B)], dim=0) # B, H, W, 2

            # For simple computation, move B to the end
            mesh = mesh.permute(-1, 1, 2, 0) # [2, H, W, B]

            # Iris wtMap
            Xi = (mesh[0, ...].squeeze() - el_iris[:, 0])*torch.cos(el_iris[:, -1])+\
                (mesh[1, ...].squeeze() - el_iris[:, 1])*torch.sin(el_iris[:, -1])
            Yi = -(mesh[0, ...].squeeze() - el_iris[:, 0])*torch.sin(el_iris[:, -1])+\
                 (mesh[1, ...].squeeze() - el_iris[:, 1])*torch.cos(el_iris[:, -1])

            # X and Y are of the shape [H, W, B]
            wtMat_iris = 1 - (Xi/el_iris[:, 2])**2 - (Yi/el_iris[:, 3])**2
            wtMat_iris = soft_heaviside(wtMat_iris, 64, mode=3)
            wtMat_iris = wtMat_iris.permute(2, 0, 1) # [B, H, W]

            # Pupil wtMap
            Xp = (mesh[0, ...].squeeze() - el_pupil[:, 0])*torch.cos(el_pupil[:, -1])+\
                (mesh[1, ...].squeeze() - el_pupil[:, 1])*torch.sin(el_pupil[:, -1])
            Yp = -(mesh[0, ...].squeeze() - el_pupil[:, 0])*torch.sin(el_pupil[:, -1])+\
                 (mesh[1, ...].squeeze() - el_pupil[:, 1])*torch.cos(el_pupil[:, -1])

            # X and Y are of the shape [H, W, B]
            wtMat_pupil = 1 - (Xp/el_pupil[:, 2])**2 - (Yp/el_pupil[:, 3])**2
            wtMat_pupil = soft_heaviside(wtMat_pupil, 64, mode=3)
            wtMat_pupil = wtMat_pupil.permute(2, 0, 1) # [B, H, W]

            # Append pupil and Iris weight maps to network skip connections
            elFeats = torch.cat([skip,
                                 wtMat_pupil.unsqueeze(1),
                                 wtMat_iris.unsqueeze(1)], dim=1)

            elFeats = self.conv1s[sc](elFeats)

            # Upsample and append maps
            featMap.append(F.interpolate(elFeats,
                                         size=(240, 320),
                                         mode='bilinear',
                                         align_corners=False))

        featMap = torch.cat(featMap, dim=1) # [B, 32, 240, 320]

        featMap = self.convStack(featMap).squeeze() # [B, 240, 320]
        featMap = torch.cat([featMap.sum(dim=1), featMap.sum(dim=2)], dim=1) # [B, 560]
        elCorr = self.linStack(featMap)

        # Additive correction
        elPred_refined = torch.clone(elPred)
        elPred_refined[:, [2, 3, 4, 7, 8, 9]] += torch.tanh(elCorr) # Between -1 and 1
        # print(F.hardtanh(elCorr))
        return elPred_refined
'''