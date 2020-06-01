#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakshit
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import normPts, linStack, convBlock
from loss import conf_Loss, get_ptLoss, get_seg2ptLoss, get_segLoss, get_selfConsistency

def getSizes(chz, growth, blks=4):
    # This function does not calculate the size requirements for head and tail

    # Encoder sizes
    sizes = {'enc': {'inter':[], 'ip':[], 'op': []},
             'dec': {'skip':[], 'ip': [], 'op': []}}
    sizes['enc']['inter'] = np.array([chz*(i+1) for i in range(0, blks)])
    sizes['enc']['op'] = np.array([np.int(growth*chz*(i+1)) for i in range(0, blks)])
    sizes['enc']['ip'] = np.array([chz] + [np.int(growth*chz*(i+1)) for i in range(0, blks-1)])

    # Decoder sizes
    sizes['dec']['skip'] = sizes['enc']['ip'][::-1] + sizes['enc']['inter'][::-1]
    sizes['dec']['ip'] = sizes['enc']['op'][::-1] #+ sizes['dec']['skip']
    sizes['dec']['op'] = np.append(sizes['enc']['op'][::-1][1:], chz)
    return sizes

class Transition_down(nn.Module):
    def __init__(self, in_c, out_c, down_size, norm, actfunc):
        super(Transition_down, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else False
        self.norm = norm(num_features=in_c)
        self.actfunc = actfunc
    def forward(self, x):
        x = self.actfunc(self.norm(x))
        x = self.conv(x)
        x = self.max_pool(x) if self.max_pool else x
        return x

class DenseNet2D_down_block(nn.Module):
    def __init__(self, in_c, inter_c, op_c, down_size, norm, actfunc):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_c+inter_c, inter_c, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(in_c+2*inter_c, inter_c, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = norm(num_features=in_c)
        self.TD = Transition_down(inter_c+in_c, op_c, down_size, norm, actfunc)

    def forward(self, x):
        x1 = self.actfunc(self.conv1(self.bn(x)))
        x21 = torch.cat([x, x1], dim=1)
        x22 = self.actfunc(self.conv22(self.conv21(x21)))
        x31 = torch.cat([x21, x22], dim=1)
        out = self.actfunc(self.conv32(self.conv31(x31)))
        out = torch.cat([out, x], dim=1)
        return out, self.TD(out)

class DenseNet2D_up_block(nn.Module):
    def __init__(self, skip_c, in_c, out_c, up_stride, actfunc):
        super(DenseNet2D_up_block, self).__init__()
        self.conv11 = nn.Conv2d(skip_c+in_c, out_c, kernel_size=1, padding=0)
        self.conv12 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(skip_c+in_c+out_c, out_c, kernel_size=1,padding=0)
        self.conv22 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.up_stride = up_stride

    def forward(self, prev_feature_map, x):
        x = F.interpolate(x,
                          mode='bilinear',
                          align_corners=False,
                          scale_factor=self.up_stride)
        x = torch.cat([x, prev_feature_map], dim=1)
        x1 = self.actfunc(self.conv12(self.conv11(x)))
        x21 = torch.cat([x, x1],dim=1)
        out = self.actfunc(self.conv22(self.conv21(x21)))
        return out

class DenseNet_encoder(nn.Module):
    def __init__(self, in_c=1, chz=32, actfunc=F.leaky_relu, growth=1.5, norm=nn.BatchNorm2d):
        super(DenseNet_encoder, self).__init__()
        sizes = getSizes(chz, growth)
        interSize = sizes['enc']['inter']
        opSize = sizes['enc']['op']
        ipSize = sizes['enc']['ip']

        self.head = convBlock(in_c=1,
                                inter_c=chz,
                                out_c=chz,
                                actfunc=actfunc)
        self.down_block1 = DenseNet2D_down_block(in_c=ipSize[0],
                                                 inter_c=interSize[0],
                                                 op_c=opSize[0],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block2 = DenseNet2D_down_block(in_c=ipSize[1],
                                                 inter_c=interSize[1],
                                                 op_c=opSize[1],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block3 = DenseNet2D_down_block(in_c=ipSize[2],
                                                 inter_c=interSize[2],
                                                 op_c=opSize[2],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block4 = DenseNet2D_down_block(in_c=ipSize[3],
                                                 inter_c=interSize[3],
                                                 op_c=opSize[3],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.bottleneck = DenseNet2D_down_block(in_c=opSize[3],
                                                 inter_c=interSize[3],
                                                 op_c=opSize[3],
                                                 down_size=0,
                                                 norm=norm,
                                                 actfunc=actfunc)
    def forward(self, x):
        x = self.head(x) # chz
        skip_1, x = self.down_block1(x) # chz
        skip_2, x = self.down_block2(x) # 2 chz
        skip_3, x = self.down_block3(x) # 4 chz
        skip_4, x = self.down_block4(x) # 8 chz
        _, x = self.bottleneck(x)
        return skip_4, skip_3, skip_2, skip_1, x

class DenseNet_decoder(nn.Module):
    def __init__(self, chz, out_c, growth, actfunc=F.leaky_relu, norm=nn.BatchNorm2d):
        super(DenseNet_decoder, self).__init__()
        sizes = getSizes(chz, growth)
        skipSize = sizes['dec']['skip']
        opSize = sizes['dec']['op']
        ipSize = sizes['dec']['ip']

        self.up_block4 = DenseNet2D_up_block(skipSize[0], ipSize[0], opSize[0], 2, actfunc)
        self.up_block3 = DenseNet2D_up_block(skipSize[1], ipSize[1], opSize[1], 2, actfunc)
        self.up_block2 = DenseNet2D_up_block(skipSize[2], ipSize[2], opSize[2], 2, actfunc)
        self.up_block1 = DenseNet2D_up_block(skipSize[3], ipSize[3], opSize[3], 2, actfunc)
        
        self.c3_11 = nn.Conv2d(kernel_size=1, in_channels=opSize[0], out_channels=3)
        self.c2_11 = nn.Conv2d(kernel_size=1, in_channels=opSize[1], out_channels=3)
        self.c1_11 = nn.Conv2d(kernel_size=1, in_channels=opSize[2], out_channels=3)

        self.final = convBlock(chz, chz, out_c, actfunc)

    def forward(self, skip4, skip3, skip2, skip1, x):
         x3 = self.up_block4(skip4, x)
         x2 = self.up_block3(skip3, x3)
         x1 = self.up_block2(skip2, x2)
         x = self.up_block1(skip1, x1)
         o = self.final(x)
         return o

class DenseNet2D(nn.Module):
    def __init__(self,
                 chz=32,
                 growth=1.2,
                 actfunc=F.leaky_relu,
                 norm=nn.InstanceNorm2d,
                 selfCorr=False,
                 disentangle=False):
        super(DenseNet2D, self).__init__()

        self.sizes = getSizes(chz, growth)

        self.toggle = True
        self.selfCorr = selfCorr
        self.disentangle = disentangle
        self.disentangle_alpha = 2

        self.enc = DenseNet_encoder(in_c=1, chz=chz, actfunc=actfunc, growth=growth, norm=norm)
        self.dec = DenseNet_decoder(chz=chz, out_c=3, actfunc=actfunc, growth=growth, norm=norm)
        self.elReg = regressionModule(self.sizes)

        self._initialize_weights()


    def setDatasetInfo(self, numSets=2):
        # Produces a 1 layered MLP which directly maps bottleneck to the DS ID
        self.numSets = numSets
        self.dsIdentify_lin = linStack(num_layers=2,
                                       in_dim=self.sizes['enc']['op'][-1],
                                       hidden_dim=64,
                                       out_dim=numSets,
                                       bias=True,
                                       actBool=False,
                                       dp=0.0)

    def forward(self,
                x, # Input batch of images [B, 1, H, W]
                target, # Target semantic output of 3 classes [B, H, W]
                pupil_center, # Pupil center [B, 2]
                elNorm, # Normalized ellipse parameters [B, 2, 5]
                spatWts, # Spatial weights for segmentation loss (boundary loss) [B, H, W]
                distMap, # Distance map for segmentation loss (surface loss) [B, 3, H, W]
                cond, # A condition array for each entry which marks its status [B, 4]
                ID, # A Tensor containing information about the dataset or subset a entry
                alpha): # Alpha score for various loss curicullum

        B, _, H, W = x.shape
        x4, x3, x2, x1, x = self.enc(x)
        latent = torch.mean(x.flatten(start_dim=2), -1) # [B, features]
        elOut = self.elReg(x, alpha) # Linear regression to ellipse parameters
        op = self.dec(x4, x3, x2, x1, x)

        #%%
        op_tup = get_allLoss(op, # Output segmentation map
                            elOut, # Predicted Ellipse parameters
                            target, # Segmentation targets
                            pupil_center, # Pupil center
                            elNorm, # Normalized ellipse equation
                            spatWts, # Spatial weights
                            distMap, # Distance maps
                            cond, # Condition
                            ID, # Image and dataset ID
                            alpha)
        
        loss, pred_c_seg = op_tup

        # Uses ellipse center from segmentation but other params from regression
        elPred = torch.cat([pred_c_seg[:, 0, :], elOut[:, 2:5],
                            pred_c_seg[:, 1, :], elOut[:, 7:10]], dim=1) # Bx5

        #%% 
        if self.selfCorr:
            loss_selfCorr = get_selfConsistency(op, elPred, 1-cond[:, 1])
            loss += 10*loss_selfCorr

        if self.disentangle:
            pred_ds = self.dsIdentify_lin(latent)
            # Disentanglement procedure
            if self.toggle:
                # Primary loss + alpha*confusion
                loss += self.disentangle_alpha*conf_Loss(pred_ds,
                                                         ID.to(torch.long),
                                                         self.toggle)
            else:
                # Secondary loss
                loss = conf_Loss(pred_ds, ID.to(torch.long), self.toggle)
        return op, elPred, latent, loss.unsqueeze(0)

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
                m.bias.data.zero_()

def get_allLoss(op, # Network output
                elOut, # Network ellipse regression output
                target, # Segmentation targets
                pupil_center, # Pupil center
                elNorm, # Normalized ellipse parameters
                spatWts,
                distMap,
                cond, # Condition matrix, 0 represents modality exists
                ID,
                alpha):

    B, C, H, W = op.shape
    loc_onlyMask = (1 -cond[:,1]).to(torch.float32) # GT mask present (True means mask exist)
    loc_onlyMask.requires_grad = False # Ensure no accidental backprop
    
    # Segmentation to pupil center loss using center of mass
    l_seg2pt_pup, pred_c_seg_pup = get_seg2ptLoss(op[:, 2, ...],
                                                  normPts(pupil_center,
                                                  target.shape[1:]), temperature=4)
    
    # Segmentation to iris center loss using center of mass
    if torch.sum(loc_onlyMask):
        # Iris center is only present when GT masks are present. Note that
        # elNorm will hold garbage values. Those samples should not be backprop
        iriMap = -op[:, 0, ...] # Inverse of background mask
        l_seg2pt_iri, pred_c_seg_iri = get_seg2ptLoss(iriMap,
                                                      elNorm[:, 0, :2],
                                                      temperature=4)
        temp = torch.stack([loc_onlyMask, loc_onlyMask], dim=1)
        l_seg2pt_iri = torch.sum(l_seg2pt_iri*temp)/torch.sum(temp.to(torch.float32))
        l_seg2pt_pup = torch.mean(l_seg2pt_pup)
        
    else:
        # If GT map is absent, loss is set to 0.0
        # Set Iris and Pupil center to be same
        l_seg2pt_iri = 0.0
        l_seg2pt_pup = torch.mean(l_seg2pt_pup)
        pred_c_seg_iri = torch.clone(elOut[:, 5:7])

    pred_c_seg = torch.stack([pred_c_seg_iri,
                              pred_c_seg_pup], dim=1) # Iris first policy
    l_seg2pt = 0.5*l_seg2pt_pup + 0.5*l_seg2pt_iri

    # Segmentation loss -> backbone loss
    l_seg = get_segLoss(op, target, spatWts, distMap, loc_onlyMask, alpha)

    # Bottleneck ellipse losses
    # NOTE: This loss is only activated when normalized ellipses do not exist
    l_pt = get_ptLoss(elOut[:, 5:7], normPts(pupil_center,
                                             target.shape[1:]), 1-loc_onlyMask)
    
    # Compute ellipse losses - F1 loss for valid samples
    l_ellipse = get_ptLoss(elOut, elNorm.view(-1, 10), loc_onlyMask)

    total_loss = l_seg2pt + 20*l_seg + 10*(l_pt + l_ellipse)

    return (total_loss, pred_c_seg)

class regressionModule(torch.nn.Module):
    def __init__(self, sizes):
        super(regressionModule, self).__init__()
        inChannels = sizes['enc']['op'][-1]
        self.max_pool = nn.AvgPool2d(kernel_size=2)

        self.c1 = nn.Conv2d(in_channels=inChannels,
                            out_channels=64,
                            bias=True,
                            kernel_size=3,
                            padding=1)

        self.c2 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            bias=True,
                            kernel_size=3,
                            padding=1)

        self.c3 = nn.Conv2d(in_channels=64+inChannels,
                            out_channels=32,
                            kernel_size=1,
                            bias=True)

        self.l1 = nn.Linear(32, 128, bias=True)
        self.l2 = nn.Linear(128, 10, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=inChannels)
        self.bn2 = nn.BatchNorm2d(num_features=64+inChannels)

        self.c_actfunc = F.hardtanh # Center has to be between -1 and 1
        self.param_actfunc = F.hardtanh # Parameters can't be negative and capped to 1

    def forward(self, x, alpha):
        # x: [B, C, 15, 20]
        B = x.shape[0]
        p = self.bn1(x)
        p = self.c1(p)
        p = self.c2(p)
        x = torch.cat([x, p], dim=1)
        x = self.bn2(x)
        x = self.c3(x)
        x = self.l1(x.reshape(B, 32, -1).sum(dim=-1))
        x = self.l2(x)

        EPS = 1e-5

        pup_c = self.c_actfunc(x[:, 0:2], min_val=-1+EPS, max_val=1-EPS)
        pup_param = self.param_actfunc(x[:, 2:4], min_val=0+EPS, max_val=1-EPS)
        pup_angle = x[:, 4]
        iri_c = self.c_actfunc(x[:, 5:7], min_val=-1+EPS, max_val=1-EPS)
        iri_param = self.param_actfunc(x[:, 7:9], min_val=0+EPS, max_val=1-EPS)
        iri_angle = x[:, 9]


        op = torch.cat([pup_c,
                        pup_param,
                        pup_angle.unsqueeze(1),
                        iri_c,
                        iri_param,
                        iri_angle.unsqueeze(1)], dim=1)
        return op