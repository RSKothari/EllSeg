"""
Created on Mon Sep  2 11:20:33 2019

@author: Shusil Dangi

References:
    https://github.com/ShusilDangi/DenseUNet-K
It is a simplied version of DenseNet with U-NET architecture.
2D implementation
"""
import math
import torch
import torch.nn as nn
from kornia.geometry.spatial_soft_argmax import SpatialSoftArgmax2d

from RITEyes_helper.utils import _NonLocalBlock2D, get_loss
from RITEyes_helper.loss_functions import ElliFit, EllipseLoss

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


class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
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

class centerPts(nn.Module):
    def __init__(self, in_channels):
        super(centerPts, self).__init__()
        self.opChannels = 2

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=int(in_channels/2),
                               kernel_size=3,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=int(in_channels/2),
                               out_channels=int(in_channels/4),
                               kernel_size=3,
                               padding=0)
        self.down = torch.nn.AdaptiveMaxPool2d((2, 2),
                                               return_indices=False)


        self.linear_1 = nn.Linear(in_features=in_channels, out_features=64)
        self.linear_2 = nn.Linear(in_features=64, out_features=4)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(self.relu(x))
        x = self.down(x)
        x = x.view(B, -1)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x.view(B, -1, 2) # [B, 4]

class DenseNet2D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=24,
                 channel_size=32,
                 concat=True,
                 dropout=False,
                 prob=0):
        super(DenseNet2D, self).__init__()

        self.nPts = int(out_channels/2)
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

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2),
                                                    dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2),
                                                    dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2),
                                                    dropout=dropout,
                                                    prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2),
                                                    dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,
                                   out_channels=out_channels + 4,
                                   kernel_size=1,
                                   padding=0)

        self.attn = _NonLocalBlock2D(in_channels=channel_size, sub_sample=False)

        self.concat = concat

        self.temperature = torch.nn.Parameter(torch.zeros(1,).cuda())
        self.softargmax = SpatialSoftArgmax2d(temperature = self.temperature,
                                              normalized_coordinates=True)
        self.distLoss = nn.SmoothL1Loss()
        self.segLoss = get_loss(nChannels=4)
        self.elFit = ElliFit(OP_pts=12, temperature=self.temperature)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, heatmaps, gtPts, Phi, spatWt, maxDist, labels, alpha, isTest):
        B, C, H, W = x.shape
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x5 = self.attn(self.x5) + self.x5
        self.x6 = self.up_block1(self.x4,self.x5)
        self.x7 = self.up_block2(self.x3,self.x6)
        self.x8 = self.up_block3(self.x2,self.x7)
        self.x9 = self.up_block4(self.x1,self.x8)
        out = self.out_conv1(self.x9)

        # Segmentation loss
        segOp = out[:, :4, ...] # Segmentation outputs
        loss_seg = self.segLoss(segOp, labels, spatWt, maxDist, alpha)

        '''
        # Centers from Segmentation masks
        centers_op, _ = self.softargmax(segOp[:, [2, 3], ...]) # Channels 2 and 3 belong to pupil/iris
        loss_center = self.distLoss(centers_op, gtPts[:, :, -1, :]) # The last point will be the centers

        # Iris points
        irisPhi, irisPts, irisMap = self.elFit(out[:, 4:(12+4), ...], gtPts[:, 0, :-1, :])

        # Pupil points
        pupilPhi, pupilPts, pupilMap = self.elFit(out[:, (12+4):, ...], gtPts[:, 1, :-1, :])

        # Ellipse points loss
        loss_pts = self.distLoss(irisPts, gtPts[:, 0, :-1, :]) + \
                    self.distLoss(pupilPts, gtPts[:, 1, :-1, :])

        # Phi loss
        loss_phi = self.distLoss(torch.stack([irisPhi, pupilPhi], dim=1),
                                 Phi)

        # Heat map loss
        loss_heat = self.distLoss(irisMap, heatmaps[:, 0, :-1, ...]) + \
                    self.distLoss(pupilMap, heatmaps[:, 1, :-1, ...])

        '''
        loss = loss_seg# + loss_heat + loss_center + loss_pts#+ loss_phi
        return segOp, loss#, irisPts, pupilPts, centers_op

