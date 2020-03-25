#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from args import parse_args
from helperfunctions import mypause
from modelSummary import model_dict
from pytorchtools import load_from_file
from torch.utils.data import DataLoader
from utils import get_nparams, get_predictions
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

#%%
if __name__ == '__main__':

    args = parse_args()

    device=torch.device("cuda")
    torch.cuda.manual_seed(12)
    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        args.useMultiGPU = False
    torch.backends.cudnn.deterministic=False

    if args.model not in model_dict:
        print("Model not found.")
        print("valid models are: {}".format(list(model_dict.keys())))
        exit(1)

    LOGDIR = os.path.join('logs', args.model, args.expname)
    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
    path2writer = os.path.join(LOGDIR, 'TB.lock')
    path2op = os.path.join('op', args.curObj)

    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
    os.makedirs(path2writer, exist_ok=True)
    os.makedirs(path2op, exist_ok=True)

    model = model_dict[args.model]

    netDict = load_from_file([args.loadfile, path2checkpoint])
    startEp = netDict['epoch'] if 'epoch' in netDict.keys() else 0
    if 'state_dict' in netDict.keys():
        model.load_state_dict(netDict['state_dict'])

    print('Parameters: {}'.format(get_nparams(model)))
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)

    f = open(os.path.join('curObjects', 'cond_'+str(args.curObj)+'.pkl'), 'rb')

    _, _, testObj = pickle.load(f)
    testObj.path2data = os.path.join(args.path2data, 'Dataset', 'All')
    testObj.augFlag = False

    testloader = DataLoader(testObj,
                            batch_size=args.batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)

    if args.disp:
        fig, axs = plt.subplots(nrows=1, ncols=1)
    #%%
    accLoss = 0.0
    imCounter = 0
    ious = []
    dists = []
    dists_seg = []
    model.eval()

    opDict = {'id':[], 'archNum': [], 'archName': [], 'code': [],
              'scores':{'iou':[], 'lat_dst':[], 'seg_dst':[]},
              'pred':{'pup_c':[], 'seg_c':[], 'mask':[]},
              'gt':{'pup_c':[], 'mask':[]}}

    with torch.no_grad():
        for bt, batchdata in enumerate(testloader):
            img, labels, spatialWeights, distMap, pupil_center, cond = batchdata
            output, latent, pred_center, seg_center, loss = model(img.to(device).to(args.prec),
                                                                  labels.to(device).long(),
                                                                  pupil_center.to(device).to(args.prec),
                                                                  spatialWeights.to(device).to(args.prec),
                                                                  distMap.to(device).to(args.prec),
                                                                  cond.to(device).to(args.prec),
                                                                  0.5)

            loss = loss if args.useMultiGPU else loss.mean()

            accLoss += loss.detach().cpu().item()
            predict = get_predictions(output)
            iou, iou_bySample = getSeg_metrics(labels.numpy(),
                                               predict.numpy(),
                                               cond.numpy())[1:]
            ptDist, ptDist_bySample = getPoint_metric(pupil_center.numpy(),
                                                      pred_center.detach().cpu().numpy(),
                                                      cond.numpy(),
                                                      img.shape[2:],
                                                      True) # Unnormalizes the points
            ptDist_seg, ptDist_seg_bySample = getPoint_metric(pupil_center.numpy(),
                                                              seg_center.detach().cpu().numpy(),
                                                              cond.numpy(),
                                                              img.shape[2:],
                                                              True) # Unnormalizes the points
            dists.append(ptDist)
            dists_seg.append(ptDist_seg)
            ious.append(iou)

            pup_c = unnormPts(pred_center.detach().cpu().numpy(),
                              img.shape[2:])
            seg_c = unnormPts(seg_center.detach().cpu().numpy(),
                              img.shape[2:])
            dispI = generateImageGrid(img.numpy(),
                                      predict.numpy(),
                                      seg_c,
                                      cond.numpy(),
                                      override=True)

            for i in range(0, img.shape[0]):
                opDict['id'].append(testObj.imList[imCounter, 0])
                opDict['archNum'].append(testObj.imList[imCounter, 1])
                opDict['archName'].append(testObj.archName[opDict['archNum']])
                opDict['code'].append(latent.detach().numpy())
                opDict['pred']['pup_c'].append(pup_c[i, :])
                opDict['pred']['seg_c'].append(seg_c[i, :])
                opDict['pred']['mask'].append(predict[i,...].numpy().astype(np.uint8))
                opDict['scores']['iou'].append(iou_bySample[i, ...])
                opDict['scores']['lat_dst'].append(ptDist_bySample[i, ...])
                opDict['scores']['seg_dst'].append(ptDist_seg_bySample[i, ...])
                opDict['gt']['pup_c'].append(pupil_center[i,...].numpy())
                opDict['gt']['mask'].append(labels[i,...].numpy().astype(np.uint8))
                imCounter+=1

            if args.disp:
                if bt == 0:
                    h_im = plt.imshow(dispI.permute(1, 2, 0))
                    plt.pause(0.01)
                else:
                    h_im.set_data(dispI.permute(1, 2, 0))
                    mypause(0.01)

        ious = np.stack(ious, axis=0)
        ious = np.nanmean(ious, axis=0)
        print('mIoU: {}. IoUs: {}'.format(np.mean(ious), ious))
        print('Latent space pupil distance: {}'.format(np.nanmean(dists),
                                                       np.nanstd(dists)))
        print('Segmentation pupil distance: {}'.format(np.mean(dists_seg),
                                                       np.nanstd(dists_seg)))

        print('--- Saving output directory ---')
        np.save(os.path.join(path2op, 'opDict.mat'), opDict)
