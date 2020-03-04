#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tqdm
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from args import parse_args
from modelSummary import model_dict
from tensorboardX import SummaryWriter
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from torch.utils.data import DataLoader
from RITEyes_helper.helperfunctions import mypause, linVal
from RITEyes_helper.pytorchtools import EarlyStopping
from RITEyes_helper.utils import get_nparams, Logger, get_predictions

#%%

def lossandaccuracy(args, loader, model, factor):
    '''
    A function to compute validation loss and performance

    Parameters
    ----------
    loader : torch loader
        Custom designed loader found in the helper functions.
    model : torch net
        Initialized model which needs to be validated againt loader.
    factor : Learning rate factor. Refer to RITNet paper for more information.
        constant.

    Returns
    -------
    TYPE
        validation score.

    '''
    epoch_loss = []
    ious = []
    model.eval()
    with torch.no_grad():
        for bt, batchdata in enumerate(loader):
            img, labels, spatialWeights, distMap, pupil_center, cond = batchdata
            output, pred_center, loss = model(img.to(device).to(args.prec),
                                                labels.to(device).long(),
                                                pupil_center.to(device).to(args.prec),
                                                spatialWeights.to(device).to(args.prec),
                                                distMap.to(device).to(args.prec),
                                                cond.to(device).to(args.prec),
                                                alpha)
            loss = loss.mean()
            epoch_loss.append(loss.item())
            
            predict = get_predictions(output)
            iou = getSeg_metrics(labels.numpy(),
                                 predict.numpy(),
                                 cond.numpy())[1]
            ious.append(iou)
    ious = np.stack(ious, axis=0)
    return np.mean(epoch_loss), np.mean(np.mean(ious, 1))

#%%
if __name__ == '__main__':

    args = parse_args()

    device=torch.device("cuda")
    torch.cuda.manual_seed(12)
    if False:#torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        args.useMultiGPU = False
    torch.backends.cudnn.deterministic=False

    if args.model not in model_dict:
        print ("Model not found.")
        print ("valid models are: {}".format(list(model_dict.keys())))
        exit(1)

    LOGDIR = os.path.join('logs', args.model, args.expname)
    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
    path2writer = os.path.join(LOGDIR, 'TB.lock')

    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
    os.makedirs(path2writer, exist_ok=True)
    
    writer = SummaryWriter(path2writer)
    logger = Logger(os.path.join(LOGDIR,'logs.log'))

    model = model_dict[args.model]

    if args.resume:
        print ("NOTE resuming training")
        model  = model.to(device)
        filename = args.loadfile
        if not os.path.exists(filename):
            print("model path not found!")
            exit(1)
        netDict = torch.load(filename)
        model.load_state_dict(netDict['state_dict'])
        startEp = netDict['epoch'] if 'epoch' in netDict.keys() else 0
    else:
        startEp = 0
    
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)
    torch.save(model.state_dict() if not args.useMultiGPU else model.module.state_dict(),
               os.path.join(path2model, args.model+'{}.pkl'.format('_init')))

    nparams = get_nparams(model)
    print('Total number of trainable parameters: {}\n'.format(nparams))

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5) # Default factor = 0.1

    patience = 10
    early_stopping = EarlyStopping(delta=1e-3,
                                   verbose=True,
                                   patience=patience,
                                   fName='checkpoint.pt',
                                   path2save=path2checkpoint)

    f = open(os.path.join('curObjects', 'cond_'+str(args.curObj)+'.pkl'), 'rb')

    trainObj, validObj, _ = pickle.load(f)
    trainObj.path2data = os.path.join(args.path2data, 'Dataset', 'All')
    validObj.path2data = os.path.join(args.path2data, 'Dataset', 'All')
    trainObj.augFlag = True
    validObj.augFlag = False

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
    
    for epoch in range(startEp, args.epochs):
        ious = []
        model.train()
        alpha = linVal(epoch, (0, args.epochs), (0, 1), 0)
        
        for bt, batchdata in enumerate(tqdm.tqdm(trainloader)):
            img, labels, spatialWeights, distMap, pupil_center, cond = batchdata
            optimizer.zero_grad()

            output, pred_center, loss = model(img.to(device).to(args.prec),
                                              labels.to(device).long(),
                                              pupil_center.to(device).to(args.prec),
                                              spatialWeights.to(device).to(args.prec),
                                              distMap.to(device).to(args.prec),
                                              cond.to(device).to(args.prec),
                                              alpha)
        
            loss = loss if args.useMultiGPU else loss.mean()
            loss.backward()
            optimizer.step()

            predict = get_predictions(output)
            iou = getSeg_metrics(labels.numpy(),
                                 predict.numpy(),
                                 cond.numpy())[1]
            ptDist = getPoint_metric(pupil_center.numpy(),
                                     pred_center.detach().cpu().numpy(),
                                     cond.numpy())
            ious.append(iou)
            
            if args.disp:
                pup_c = unnormPts(pred_center.detach().cpu().numpy(),
                                  img.shape[2:])
                dispI = generateImageGrid(img.numpy(),
                                          predict.numpy(),
                                          pup_c,
                                          cond.numpy())
                if (epoch == 0) and (bt == 0):
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
        ious = np.mean(ious, axis=0)
        logger.write('Epoch:{}, Train mIoU: {}'.format(epoch, ious))
        lossvalid, miou = lossandaccuracy(args, validloader, model, alpha)
        
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {}'
        logger.write(f.format(epoch, lossvalid, miou))

        scheduler.step(lossvalid)
        early_stopping(lossvalid, model.state_dict() if not args.useMultiGPU else model.module.state_dict())
        
        if early_stopping.early_stop:
            netDict = {'state_dict':[], 'epoch': epoch}
            netDict['state_dict'] = model.state_dict() if not args.useMultiGPU else model.module.state_dict()
            torch.save(netDict, os.path.join(path2model, args.model + 'earlystop_{}.pkl'.format(epoch)))
            print("Early stopping")
            break

        ##save the model every epoch
        if epoch %5 == 0:
            torch.save(model.state_dict() if not args.useMultiGPU else model.module.state_dict(),
                       os.path.join(path2model, args.model+'_{}.pkl'.format(epoch)))
    writer.close()