#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:42 2020

@author: aayush
This is a pytorch implementation of deepvog
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import get_seg2ptLoss
from utils import normPts

class encoding_block(nn.Module):
#    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
    def __init__(self, input_channels, filter_size, filters_num, layer_num, block_type, stage, s = 1, X_skip=0):
        super(encoding_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,filters_num,kernel_size=filter_size,stride=(s,s),padding=(1,1)) #same
        self.conv2 = nn.Conv2d(filters_num,filters_num*2,kernel_size=(2,2),stride=(2,2),padding=(0,0))   #valid
        self.relu = nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(num_features=filters_num)
        self.bn2 = torch.nn.BatchNorm2d(num_features=filters_num*2)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x_down=self.conv2(x)
        x_down=self.bn2(x_down)
        x_down=self.relu(x_down)
        return x,x_down


class decoding_block(nn.Module):
#    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
    def __init__(self, skip_channels, input_channels, filter_size, filters_num, layer_num, block_type, stage, s = 1, up_stride=(2,2),X_jump=0, up_sampling = True):
        super(decoding_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels+skip_channels,filters_num,kernel_size=filter_size,stride=(s,s),padding=(1,1)) #same
        self.conv2 = nn.Conv2d(filters_num,filters_num,kernel_size=filter_size,stride=(1,1),padding=(1,1))   #valid
        self.relu = nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(num_features=filters_num)
        self.bn2 = torch.nn.BatchNorm2d(num_features=filters_num)
        self.X_jump=X_jump ##No need of this as prev_feature map is given as None when no concatenation is needed
        self._initialize_weights()
        self.up_sampling=up_sampling
        self.up_stride=up_stride

    def _initialize_weights(self):
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self,prev_feature_map, x):
        if prev_feature_map is not None:
            # print (x.shape, prev_feature_map.shape)
            x = torch.cat((x,prev_feature_map),dim=1)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        if self.up_sampling:
            # print ('here')
            # print (x.shape)
            x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
            # print (x.shape)
            x=self.conv2(x)
            x=self.bn2(x)
            x=self.relu(x)
            # print (x.shape)
        return x


class DeepVOG_pytorch(nn.Module):
    def __init__(self,in_channels=3,out_channels=2,filter_size=(3,3)):
        super(DeepVOG_pytorch, self).__init__()

        self.output_channels=16
        self.down_block1 = encoding_block(input_channels=in_channels,filter_size=filter_size,\
                        filters_num=self.output_channels, layer_num=1, block_type='down', stage=1, s=1)
        self.down_block2 = encoding_block(input_channels=self.output_channels*2,filter_size=filter_size,\
                        filters_num=self.output_channels*2, layer_num=1, block_type='down', stage=1, s=1)
        self.down_block3 = encoding_block(input_channels=self.output_channels*4,filter_size=filter_size,\
                        filters_num=self.output_channels*4, layer_num=1, block_type='down', stage=1, s=1)
        self.down_block4 = encoding_block(input_channels=self.output_channels*8,filter_size=filter_size,\
                        filters_num=self.output_channels*8, layer_num=1, block_type='down', stage=1, s=1)

        self.up_block1 = decoding_block(skip_channels=0,input_channels=self.output_channels*16,filter_size=filter_size,\
                        filters_num=self.output_channels*16, layer_num=1, block_type='up', stage=1, s=1)
        self.up_block2 = decoding_block(skip_channels=self.output_channels*8,input_channels=self.output_channels*16,filter_size=filter_size,\
                        filters_num=self.output_channels*16, layer_num=1, block_type='up', stage=1, s=1)
        self.up_block3 = decoding_block(skip_channels=self.output_channels*4,input_channels=self.output_channels*16,filter_size=filter_size,\
                        filters_num=self.output_channels*8, layer_num=1, block_type='up', stage=1, s=1)
        self.up_block4 = decoding_block(skip_channels=self.output_channels*2,input_channels=self.output_channels*8,filter_size=filter_size,\
                        filters_num=self.output_channels*4, layer_num=1, block_type='up', stage=1, s=1)
        self.up_block5 = decoding_block(skip_channels=self.output_channels,input_channels=self.output_channels*4,filter_size=filter_size,\
                        filters_num=self.output_channels*2, layer_num=1, block_type='up', stage=1, s=1, up_sampling=False)
    # Output layer operations
        self.conv1 = nn.Conv2d(self.output_channels*2,out_channels,kernel_size=(1,1),stride=(1,1),padding=(0,0)) #same
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

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
        B = x.shape[0]
        x = torch.cat([x, x, x], dim=1) # For compatibility with original code
        X_Jump1,self.x1 = self.down_block1(x)  ##32
        X_Jump2,self.x2 = self.down_block2(self.x1) ##64
        X_Jump3,self.x3 = self.down_block3(self.x2) ##128
        X_Jump4,self.x4 = self.down_block4(self.x3) ##256
        self.x5 = self.up_block1(None,self.x4) ##256
        self.x6 = self.up_block2(X_Jump4,self.x5) ##128
        self.x7 = self.up_block3(X_Jump3,self.x6) ##64
        self.x8 = self.up_block4(X_Jump2,self.x7) ##32
        self.x9 = self.up_block5(X_Jump1,self.x8) ##32
        self.x =self.conv1(self.x9)
        out = self.x
        
        loss, pred_c_seg_pup = get_allLoss(out, target, pupil_center, cond)
        elPred = torch.cat([pred_c_seg_pup, torch.rand(B,3).to(pred_c_seg_pup.device),
                            pred_c_seg_pup, torch.rand(B,3).to(pred_c_seg_pup.device)],
                           dim=1) # Bx5
        embedding = torch.ones(B, 5) # Garbage
        return out, elPred, embedding, loss.unsqueeze(0)

def get_allLoss(op, # Network output
                target, # Segmentation targets
                pupil_center, # Pupil center
                cond, # Condition
                ):
    B = op.shape[0]
    log_seg_ok = 1-cond[:, 1]
    target = (target == 2).to(torch.long) # 1 for pupil rest 0
    l_seg2pt_pup, pred_c_seg_pup = get_seg2ptLoss(op[:, 1, ...],
                                                  normPts(pupil_center,
                                                          target.shape[1:]), temperature=4)
    l_seg = 10*F.cross_entropy(torch.softmax(op, dim=1), target, reduction='none')
    if torch.sum(log_seg_ok):
        l_seg = torch.sum(l_seg.reshape(B, -1).mean(dim=1)*log_seg_ok)/torch.sum(log_seg_ok)
    else:
        l_seg = 0.0
    loss = l_seg + torch.mean(l_seg2pt_pup)
    return loss, pred_c_seg_pup