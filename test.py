#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tqdm
import torch
import pickle
import resource
import numpy as np
import matplotlib.pyplot as plt

from args import parse_args
from modelSummary import model_dict
from pytorchtools import load_from_file
from torch.utils.data import DataLoader

from helperfunctions import mypause, stackall_Dict

from loss import get_seg2ptLoss

from utils import get_nparams, get_predictions
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*10, rlimit[1]))

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

    LOGDIR = os.path.join(os.getcwd(), 'logs', args.model, args.expname)
    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
    path2writer = os.path.join(LOGDIR, 'TB.lock')
    path2op = os.path.join(os.getcwd(), 'op', str(args.curObj))

    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
    os.makedirs(path2writer, exist_ok=True)
    os.makedirs(path2op, exist_ok=True)

    model = model_dict[args.model]

    netDict = load_from_file([args.loadfile,
                              os.path.join(path2checkpoint, 'checkpoint.pt')])
    startEp = netDict['epoch'] if 'epoch' in netDict.keys() else 0
    if 'state_dict' in netDict.keys():
        model.load_state_dict(netDict['state_dict'])

    print('Parameters: {}'.format(get_nparams(model)))
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)

    f = open(os.path.join('curObjects',
                          'baseline',
                          'cond_'+str(args.curObj)+'.pkl'), 'rb')

    _, _, testObj = pickle.load(f)
    testObj.path2data = os.path.join(args.path2data, 'Datasets', 'All')
    testObj.augFlag = False

    testloader = DataLoader(testObj,
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)

    if args.disp:
        fig, axs = plt.subplots(nrows=1, ncols=1)
    #%%
    accLoss = 0.0
    imCounter = 0

    ious = []

    dists_pupil_latent = []
    dists_pupil_seg = []

    dists_iris_latent = []
    dists_iris_seg = []

    model.eval()

    opDict = {'id':[], 'archNum': [], 'archName': [], 'code': [],
              'scores':{'iou':[], 'lat_dst':[], 'seg_dst':[]},
              'pred':{'pup_latent_c':[],
                      'pup_seg_c':[],
                      'iri_latent_c':[],
                      'iri_seg_c':[],
                      'mask':[]},
              'gt':{'pup_c':[], 'mask':[]}}

    with torch.no_grad():
        for bt, batchdata in enumerate(tqdm.tqdm(testloader)):
            img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = batchdata
            out_tup = model(img.to(device).to(args.prec),
                            labels.to(device).long(),
                            pupil_center.to(device).to(args.prec),
                            elNorm.to(device).to(args.prec),
                            spatialWeights.to(device).to(args.prec),
                            distMap.to(device).to(args.prec),
                            cond.to(device).to(args.prec),
                            imInfo[:, 2].to(device).to(torch.long),
                            0.5)

            output, elOut, latent, loss = out_tup

            latent_pupil_center = elOut[:, 0:2].detach().cpu().numpy()
            latent_iris_center  = elOut[:, 5:7].detach().cpu().numpy()

            _, seg_pupil_center = get_seg2ptLoss(output[:, 2, ...].cpu(), pupil_center, temperature=4)
            _, seg_iris_center  = get_seg2ptLoss(-output[:, 0, ...].cpu(), iris_center, temperature=4)

            loss = loss if args.useMultiGPU else loss.mean()

            accLoss += loss.detach().cpu().item()
            predict = get_predictions(output)

            iou, iou_bySample = getSeg_metrics(labels.numpy(),
                                               predict.numpy(),
                                               cond[:, 1].numpy())[1:]

            latent_pupil_dist, latent_pupil_dist_bySample = getPoint_metric(pupil_center.numpy(),
                                                                            latent_pupil_center,
                                                                            cond[:,0].numpy(),
                                                                            img.shape[2:],
                                                                            True) # Unnormalizes the points

            seg_pupil_dist, seg_pupil_dist_bySample = getPoint_metric(pupil_center.numpy(),
                                                                      seg_pupil_center,
                                                                      cond[:,1].numpy(),
                                                                      img.shape[2:],
                                                                      True) # Unnormalizes the points

            latent_iris_dist, latent_iris_dist_bySample = getPoint_metric(iris_center.numpy(),
                                                                          latent_iris_center,
                                                                          cond[:,1].numpy(),
                                                                          img.shape[2:],
                                                                          True) # Unnormalizes the points

            seg_iris_dist, seg_iris_dist_bySample = getPoint_metric(iris_center.numpy(),
                                                                    seg_iris_center,
                                                                    cond[:,1].numpy(),
                                                                    img.shape[2:],
                                                                    True) # Unnormalizes the points

            dists_pupil_latent.append(latent_pupil_dist)
            dists_iris_latent.append(latent_iris_dist)
            dists_pupil_seg.append(seg_pupil_dist)
            dists_iris_seg.append(seg_iris_dist)

            ious.append(iou)

            pup_latent_c = unnormPts(latent_pupil_center,
                                     img.shape[2:])
            pup_seg_c = unnormPts(seg_pupil_center,
                                  img.shape[2:])
            iri_latent_c = unnormPts(latent_iris_center,
                                     img.shape[2:])
            iri_seg_c = unnormPts(seg_iris_center,
                                  img.shape[2:])

            dispI = generateImageGrid(img.numpy().squeeze(),
                                      predict.numpy(),
                                      elOut.detach().cpu().numpy().reshape(-1, 2, 5),
                                      pup_seg_c,
                                      cond.numpy(),
                                      override=True,
                                      heatmaps=False)

            for i in range(0, img.shape[0]):
                archNum = testObj.imList[imCounter, 1]
                opDict['id'].append(testObj.imList[imCounter, 0])
                opDict['code'].append(latent[i,...].detach().cpu().numpy())

                opDict['archNum'].append(archNum)
                opDict['archName'].append(testObj.arch[archNum])

                opDict['pred']['pup_latent_c'].append(pup_latent_c[i, :])
                opDict['pred']['pup_seg_c'].append(pup_seg_c[i, :])
                opDict['pred']['iri_latent_c'].append(iri_latent_c[i, :])
                opDict['pred']['iri_seg_c'].append(iri_seg_c[i, :])

                if args.test_save_op_masks:
                    opDict['pred']['mask'].append(predict[i,...].numpy().astype(np.uint8))

                opDict['scores']['iou'].append(iou_bySample[i, ...])
                opDict['scores']['lat_dst'].append(latent_pupil_dist_bySample[i, ...])
                opDict['scores']['seg_dst'].append(seg_pupil_dist_bySample[i, ...])

                opDict['gt']['pup_c'].append(pupil_center[i,...].numpy())

                if args.test_save_op_masks:
                    opDict['gt']['mask'].append(labels[i,...].numpy().astype(np.uint8))

                imCounter+=1

            if args.disp:
                if bt == 0:
                    h_im = plt.imshow(dispI.permute(1, 2, 0))
                    plt.pause(0.01)
                else:
                    h_im.set_data(dispI.permute(1, 2, 0))
                    mypause(0.01)

        opDict = stackall_Dict(opDict)
        ious = np.stack(ious, axis=0)
        ious = np.nanmean(ious, axis=0)
        print('mIoU: {}. IoUs: {}'.format(np.mean(ious), ious))
        print('Latent space PUPIL dist. Med: {}, STD: {}'.format(np.nanmedian(dists_pupil_latent),
                                                            np.nanstd(dists_pupil_latent)))
        print('Segmentation PUPIL dist. Med: {}, STD: {}'.format(np.nanmedian(dists_pupil_seg),
                                                            np.nanstd(dists_pupil_seg)))
        print('Latent space IRIS dist. Med: {}, STD: {}'.format(np.nanmedian(dists_iris_latent),
                                                           np.nanstd(dists_iris_latent)))
        print('Segmentation IRIS dist. Med: {}, STD: {}'.format(np.nanmedian(dists_iris_seg),
                                                           np.nanstd(dists_iris_seg)))

        print('--- Saving output directory ---')
        f = open(os.path.join(path2op, 'opDict.pkl'), 'wb')
        pickle.dump(opDict, f)
        f.close()
