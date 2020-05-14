#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:10:53 2020

@author: aayush
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakshit
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import normPts, regressionModule, linStack
from loss import conf_Loss, get_ptLoss, get_seg2ptLoss, get_segLoss, get_seg2elLoss, get_selfConsistency

def getSizes(chz, growth, blks=4):
    # This function does not calculate the size requirements for head and tail

    # Encoder sizes
    sizes = {'enc': {'inter':[], 'ip':[], 'op': []},
             'dec': {'skip':[], 'ip': [], 'op': []}}
    sizes['enc']['inter'] = np.array([chz for i in range(0, blks)])
    sizes['enc']['op'] = np.array([chz for i in range(0, blks)])
    sizes['enc']['ip'] = np.array([chz for i in range(0, blks)])

    # Decoder sizes
    sizes['dec']['skip'] = sizes['enc']['ip'][::-1] + sizes['enc']['inter'][::-1]
    sizes['dec']['ip'] = sizes['enc']['op'][::-1] #+ sizes['dec']['skip']
    sizes['dec']['op'] = np.append(sizes['enc']['op'][::-1][1:], chz)
    return sizes

class DenseNet2D_down_block(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out)

class DenseNet2D_up_block(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride=2,dropout=False,prob=0):
        super(DenseNet2D_up_block, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels+input_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
                                kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
        return out

class DenseNet_encoder(nn.Module):
    def __init__(self, in_channels=1,
                 out_channels=3,
                 channel_size=32,
                 actfunc=F.leaky_relu,
                 norm=nn.BatchNorm2d,
                 concat=True,
                 dropout=False,
                 prob=0):
        super(DenseNet_encoder, self).__init__()
   
        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,
                                                 output_channels=channel_size,
                                                 down_size=None,dropout=dropout,
                                                 prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size,
                                                 output_channels=channel_size,
                                                 down_size=(2,2),
                                                 dropout=dropout,
                                                 prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,
                                                 output_channels=channel_size,
                                                 down_size=(2,2),
                                                 dropout=dropout,
                                                 prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,
                                                 output_channels=channel_size,
                                                 down_size=(2,2),
                                                 dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,
                                                 output_channels=channel_size,
                                                 down_size=(2,2),
                                                 dropout=dropout,
                                                 prob=prob)
        
    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        return self.x4,self.x3,self.x2,self.x1,self.x5 
    
class DenseNet_decoder(nn.Module):
    def __init__(self, in_channels=1,
                 out_channels=3,
                 channel_size=32,
                 actfunc=F.leaky_relu,
                 norm=nn.BatchNorm2d,
                 concat=True,
                 dropout=False,
                 prob=0):
        super(DenseNet_decoder, self).__init__()

        self.up_block1 = DenseNet2D_up_block(skip_channels=channel_size,
                                             input_channels=channel_size,
                                             output_channels=channel_size,
                                             up_stride=(2,2),
                                             dropout=dropout,
                                             prob=prob)
        self.up_block2 = DenseNet2D_up_block(skip_channels=channel_size,
                                             input_channels=channel_size,
                                             output_channels=channel_size,
                                             up_stride=(2,2),
                                             dropout=dropout,
                                             prob=prob)
        self.up_block3 = DenseNet2D_up_block(skip_channels=channel_size,
                                             input_channels=channel_size,
                                             output_channels=channel_size,
                                             up_stride=(2,2),
                                             dropout=dropout,
                                             prob=prob)
        self.up_block4 = DenseNet2D_up_block(skip_channels=channel_size,
                                             input_channels=channel_size,
                                             output_channels=channel_size,
                                             up_stride=(2,2),
                                             dropout=dropout,
                                             prob=prob)

        self.final = nn.Conv2d(in_channels=channel_size,
                               out_channels=out_channels,
                               kernel_size=1,
                               padding=0)        

    def forward(self, skip4, skip3, skip2, skip1, x):
         x = self.up_block4(skip4, x)
         x = self.up_block3(skip3, x)
         x = self.up_block2(skip2, x)
         x = self.up_block1(skip1, x)
         o = self.final(x)
         return o

class DenseNet2D(nn.Module):
    def __init__(self,
                 chz=32,
                 growth=1.2,
                 actfunc=F.leaky_relu,
                 norm=nn.BatchNorm2d,
                 selfCorr=False,
                 disentangle=False,
                 dropout=True,
                 prob=0.2):
        super(DenseNet2D, self).__init__()

        self.sizes = getSizes(chz, growth)

        self.toggle = True
        self.selfCorr = selfCorr
        self.disentangle = disentangle
        self.disentangle_alpha = 2

        self.enc = DenseNet_encoder(in_channels=1, 
                                    out_channels=3, 
                                    channel_size=chz,
                                    actfunc=actfunc, 
                                    norm=norm,
                                    concat=True,
                                    dropout=False,
                                    prob=0)
        self.dec = DenseNet_decoder(in_channels=1, 
                                    out_channels=3, 
                                    channel_size=chz,
                                    actfunc=actfunc, 
                                    norm=norm,
                                    concat=True,
                                    dropout=False,
                                    prob=0)
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
        
        # Segmentation to ellipse loss
        loss_seg2el = get_seg2elLoss(target==2, elPred[:, 5:], 1-cond[:,1]) +\
                        get_seg2elLoss(~(target==0), elPred[:, :5], 1-cond[:,1])
        loss += loss_seg2el
        
        if self.selfCorr:
            loss += 10*get_selfConsistency(op, elPred, 1-cond[:, 1])

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