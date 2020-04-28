#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from args import parse_args
from modelSummary import model_dict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from helperfunctions import mypause, linVal
from pytorchtools import EarlyStopping, load_from_file
from utils import get_nparams, Logger, get_predictions, lossandaccuracy
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts
from utils import points_to_heatmap, getAng_metric

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

#%%
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Deactive file locking
embed_log = 5
EPS=1e-7

if __name__ == '__main__':

    args = parse_args()

    device=torch.device("cuda")
    torch.cuda.manual_seed(12)
    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        print('Single GPU setup')
        args.useMultiGPU = False
    torch.backends.cudnn.deterministic=True

    if args.model not in model_dict:
        print ("Model not found.")
        print ("valid models are: {}".format(list(model_dict.keys())))
        exit(1)

    LOGDIR = os.path.join(os.getcwd(), 'logs', args.model, args.expname)
    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
    path2writer = os.path.join(LOGDIR, 'TB.lock')

    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
    os.makedirs(path2writer, exist_ok=True)

    f = open(os.path.join('curObjects','baseline','cond_'+str(args.curObj)+'.pkl'), 'rb')

    trainObj, validObj, _ = pickle.load(f)
    trainObj.path2data = os.path.join(args.path2data, 'Dataset', 'All')
    validObj.path2data = os.path.join(args.path2data, 'Dataset', 'All')
    trainObj.augFlag = True
    validObj.augFlag = False

    writer = SummaryWriter(path2writer)
    logger = Logger(os.path.join(LOGDIR,'logs.log'))

    # Ensure model has all necessary weights initialized
    model = model_dict[args.model]
    model.selfCorr = args.selfCorr
    model.disentangle = args.disentangle

    param_list = [param for name, param in model.named_parameters() if 'dsIdentify' not in name]
    optimizer = torch.optim.Adam([{'params':param_list,
                                   'lr':args.lr}]) # Set optimizer

    # If loading pretrained weights, ensure you don't load confusion branch
    if args.resume:
        print ("NOTE resuming training. Priority: 1) Checkpoint 2) Epoch #")
        model  = model.to(device)
        checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')
        netDict = load_from_file([checkpointfile, args.loadfile])
        model.load_state_dict(netDict['state_dict'])
        startEp = netDict['epoch'] if 'epoch' in netDict.keys() else 0
    else:
        # If the very first epoch, then save out an _init pickle
        # This is particularly useful for lottery tickets
        startEp = 0
        torch.save(model.state_dict(), os.path.join(path2model, args.model+'{}.pkl'.format('_init')))

    # Let the network know you need a disentanglement module.
    # Please refer to args.py for more information on disentanglement strategy
    if args.disentangle:
        # Let the model know how many datasets it must expect
        print('Total # of datasets found: {}'.format(np.unique(trainObj.imList[:, 2]).size))
        model.setDatasetInfo(np.unique(trainObj.imList[:, 2]).size)
        opt_disent = torch.optim.Adam(model.dsIdentify_lin.parameters(), lr=1*args.lr)

    nparams = get_nparams(model)
    print('Total number of trainable parameters: {}\n'.format(nparams))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max',
                                                           patience=5,
                                                           verbose=True,
                                                           factor=0.01) # Default factor = 0.1

    patience = 5
    early_stopping = EarlyStopping(mode='max',
                                   delta=0.01,
                                   verbose=True,
                                   patience=patience,
                                   fName='checkpoint.pt',
                                   path2save=path2checkpoint)

    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec) # NOTE: good habit to do this before optimizer

    if args.overfit > 0:
        # This is a flag to check if attempting to overfit
        trainObj.imList = trainObj.imList[:args.overfit*args.batchsize,:]
        validObj.imList = validObj.imList[:args.overfit*args.batchsize,:]

    trainloader = DataLoader(trainObj,
                             batch_size=args.batchsize,
                             shuffle=True,
                             num_workers=args.workers,
                             drop_last=True)
    validloader = DataLoader(validObj,
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=args.workers,
                             drop_last=True)

    if args.disp:
        fig, axs = plt.subplots(nrows=1, ncols=1)
    #%%
    for epoch in range(startEp, args.epochs):
        accLoss = 0.0
        ious = []
        pup_c_lat_dists = []
        pup_c_seg_dists = []
        pup_ang_lat = []

        model.train()
        alpha = linVal(epoch, (0, args.epochs), (0, 1), 0)

        for bt, batchdata in enumerate(trainloader):
            img, labels, spatialWeights, distMap, pupil_center, elPts, elNorm, cond, imInfo = batchdata
            hMaps = points_to_heatmap(elPts, 2, img.shape[2:])

            model.toggle = False
            optimizer.zero_grad()

            # Disentanglement procedure. Toggle should always be False upon entry.
            if args.disentangle:
                for name, param in model.named_parameters():
                    # Freeze unrequired weights
                    if 'dsIdentify_lin' not in name:
                        # Freeze all unnecessary weights
                        param.requires_grad=False
                    else:
                        param.requires_grad=True

                val = 100 # Random large value
                while not model.toggle:
                    # Keep forward passing until secondary is finetuned
                    opt_disent.zero_grad()
                    out_tup = model(img.to(device).to(args.prec),
                                    labels.to(device).long(),
                                    pupil_center.to(device).to(args.prec),
                                    hMaps.to(device).to(args.prec),
                                    elPts.to(device).to(args.prec),
                                    elNorm.to(device).to(args.prec),
                                    spatialWeights.to(device).to(args.prec),
                                    distMap.to(device).to(args.prec),
                                    cond.to(device).to(args.prec),
                                    imInfo[:, 2].to(device).to(torch.long), # Send DS #
                                    alpha)
                    output, elOut, _, pred_center, seg_center, loss = out_tup
                    loss = loss.mean() if args.useMultiGPU else loss
                    loss.backward()
                    opt_disent.step()

                    diff = val - loss.detach().item() # Loss derivative
                    val = loss.detach().item() # Update previous loss value
                    model.toggle = True if diff < EPS else False

                # Switch the parameters which requires gradients
                for name, param in model.named_parameters():
                    param.requires_grad = False if 'dsIdentify_lin' in name else True

            model.toggle = True # This must always be true to optimize primary + conf loss
            out_tup = model(img.to(device).to(args.prec),
                            labels.to(device).long(),
                            pupil_center.to(device).to(args.prec),
                            hMaps.to(device).to(args.prec),
                            elPts.to(device).to(args.prec),
                            elNorm.to(device).to(args.prec),
                            spatialWeights.to(device).to(args.prec),
                            distMap.to(device).to(args.prec),
                            cond.to(device).to(args.prec),
                            imInfo[:, 2].to(device).to(torch.long), # Send DS #
                            alpha)
            output, elOut, _, pred_center, seg_center, loss = out_tup

            loss = loss.mean() if args.useMultiGPU else loss
            loss.backward()
            optimizer.step()

            accLoss += loss.detach().cpu().item()
            predict = get_predictions(output)
            iou = getSeg_metrics(labels.numpy(),
                                 predict.numpy(),
                                 cond[:, 1].numpy())[1]
            ptDist = getPoint_metric(pupil_center.numpy(),
                                     pred_center.detach().cpu().numpy(),
                                     cond[:,0].numpy(),
                                     img.shape[2:],
                                     True)[0] # Unnormalizes the points
            ptDist_seg = getPoint_metric(pupil_center.numpy(),
                                         seg_center.detach().cpu().numpy(),
                                         cond[:,0].numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points
            angDist_reg = getAng_metric(elNorm[:, 1, -1].numpy(),
                                        elOut[:, -1].detach().cpu().numpy(),
                                        cond[:, 1].numpy())[0]

            # Append scores to running metric
            pup_c_lat_dists.append(ptDist)
            pup_c_seg_dists.append(ptDist_seg)
            pup_ang_lat.append(angDist_reg)
            ious.append(iou)

            pup_c = unnormPts(pred_center.detach().cpu().numpy(),
                              img.shape[2:])
            seg_c = unnormPts(seg_center.detach().cpu().numpy(),
                              img.shape[2:])
            dispI = generateImageGrid(img.squeeze().numpy(),
                                      predict.numpy(),
                                      hMaps.detach().cpu().numpy(),
                                      elOut.detach().cpu().numpy().reshape(-1, 2, 5),
                                      pup_c,
                                      cond.numpy(),
                                      override=True,
                                      heatmaps=False)

            if args.disp:
                if (epoch == startEp) and (bt == 0):
                    h_im = plt.imshow(dispI.permute(1, 2, 0))
                    plt.pause(0.01)
                else:
                    h_im.set_data(dispI.permute(1, 2, 0))
                    mypause(0.01)

            if bt%10 == 0:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch,
                                                                     bt,
                                                                     len(trainloader),
                                                                     loss.item()))

        ious = np.stack(ious, axis=0)
        ious = np.nanmean(ious, axis=0)
        logger.write('Epoch:{}, Train IoU: {}'.format(epoch, ious))

        # Add info to tensorboard
        writer.add_scalar('train/loss', accLoss/bt, epoch)
        writer.add_scalars('train/pup_l_c', {'mu':np.nanmean(pup_c_lat_dists),
                                             'std':np.nanstd(pup_c_lat_dists)}, epoch)
        writer.add_scalars('train/pup_s_c', {'mu':np.nanmean(pup_c_seg_dists),
                                             'std':np.nanstd(pup_c_seg_dists)}, epoch)
        if np.any(~np.isnan(pup_ang_lat)):
            # These values will not exist when training with pure pupil center code
            writer.add_scalars('train/pup_ang', {'mu':np.nanmean(pup_ang_lat),
                                                 'std':np.nanstd(pup_ang_lat)}, epoch)
            writer.add_scalars('train/iou', {'mIOU':np.mean(ious),
                                             'bG':ious[0],
                                             'iris':ious[1],
                                             'pupil':ious[2]}, epoch)

        out_tup = lossandaccuracy(args, # Training arguments
                                  validloader, # Validation loader
                                  model, # Model
                                  alpha, # Alpha value to measure loss
                                  device)
        lossvalid, ious, pup_c_lat_dists, pup_c_seg_dists, pup_ang_lat, latent_codes = out_tup

        # Add valid info to tensorboard
        writer.add_scalar('valid/loss', lossvalid, epoch)
        writer.add_scalars('valid/pup_l_c', {'mu':np.nanmean(pup_c_lat_dists),
                                             'std':np.nanstd(pup_c_lat_dists)}, epoch)
        writer.add_scalars('valid/pup_s_c', {'mu':np.nanmean(pup_c_seg_dists),
                                             'std':np.nanstd(pup_c_seg_dists)}, epoch)
        if np.any(~np.isnan(pup_ang_lat)):
            # These values will not exist when training with pure pupil center code
            writer.add_scalars('valid/pup_ang', {'mu':np.nanmean(pup_ang_lat),
                                                 'std':np.nanstd(pup_ang_lat)}, epoch)
            writer.add_scalars('valid/iou', {'mIOU':np.mean(ious),
                                             'bG':ious[0],
                                             'iris':ious[1],
                                             'pupil':ious[2]}, epoch)
        writer.add_image('train/op', dispI, epoch)

        if epoch%embed_log == 0:
            print('Saving embeddings ...')
            writer.add_embedding(torch.cat(latent_codes, 0),
                                 metadata=validObj.imList[:len(latent_codes)*args.batchsize, 2],
                                 global_step=epoch)

        f = 'Epoch:{}, Valid Loss: {:.3f}, mIoU: {}'
        logger.write(f.format(epoch, lossvalid, np.mean(ious)))

        # Generate a model dictionary which stores epochs and current state
        netDict = {'state_dict':[], 'epoch': epoch}
        stateDict = model.state_dict() if not args.useMultiGPU else model.module.state_dict()
        netDict['state_dict'] = {k: v for k, v in stateDict.items() if 'dsIdentify_lin' not in k}

        if not np.isnan(np.mean(ious)):
            scoreTracker = np.mean(ious) + 2 - 2.5e-3*(np.nanmean(pup_c_lat_dists) + np.nanmean(pup_c_seg_dists)) # Max value 3
        else:
            scoreTracker = 2 - 2.5e-3*(np.nanmean(pup_c_lat_dists) + np.nanmean(pup_c_seg_dists)) # Max value 2
        scheduler.step(scoreTracker)
        early_stopping(scoreTracker, netDict)

        if early_stopping.early_stop:
            torch.save(netDict, os.path.join(path2model, args.model + 'earlystop_{}.pkl'.format(epoch)))
            print("Early stopping")
            break

        ##save the model every 2 epochs
        if epoch%2 == 0:
            torch.save(netDict,
                       os.path.join(path2model, args.model+'_{}.pkl'.format(epoch)))
    writer.close()
