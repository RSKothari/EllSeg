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
from utils import get_nparams, get_predictions
from helperfunctions import mypause, stackall_Dict
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts
from utils import getAng_metric,Logger

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*10, rlimit[1]))

#%%
if __name__ == '__main__':

    args = parse_args()
#%%
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

    if args.seg2elactivated:
        path_intermediate='with_seg2el'
    else:
        path_intermediate='without_seg2el'       

    if args.expname=='':
        args.expname='RC_e2e_'+args.model+'_'+args.curObj+'_0_0'

    LOGDIR = os.path.join(os.getcwd(), 'ExpData',path_intermediate, 'logs',\
                          args.model, args.expname)
#    LOGDIR = os.path.join(os.getcwd(), 'logs', args.model, args.expname)
#    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
#    path2writer = os.path.join(LOGDIR, 'TB.lock')
    path2op = os.path.join(os.getcwd(), 'op', str(args.curObj),args.model)
    path2op_mask = os.path.join(os.getcwd(), 'op', str(args.curObj), args.model,'mask'+path_intermediate)

#%%
    os.makedirs(LOGDIR, exist_ok=True)
#    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
#    os.makedirs(path2writer, exist_ok=True)
    os.makedirs(path2op, exist_ok=True)
    os.makedirs(path2op_mask, exist_ok=True)

    model = model_dict[args.model]

    checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')
    netDict = load_from_file([checkpointfile, args.loadfile])
    model.load_state_dict(netDict['state_dict'])
    
#    netDict = load_from_file([args.loadfile, path2checkpoint])
#    startEp = netDict['epoch'] if 'epoch' in netDict.keys() else 0
#    if 'state_dict' in netDict.keys():
#        model.load_state_dict(netDict['state_dict'])

    print('Parameters: {}'.format(get_nparams(model)))
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)

    f = open(os.path.join(os.getcwd(),'curObjects', 'baseline/cond_'+str(args.curObj)+'.pkl'), 'rb')

    _, _, testObj = pickle.load(f)
    testObj.path2data = os.path.join(args.path2data, 'Dataset', 'All')
    testObj.augFlag = False

    testloader = DataLoader(testObj,
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)

#    if args.disp:
#        fig, axs = plt.subplots(nrows=1, ncols=1)
    #%%
    accLoss = 0.0
    imCounter = 0
    ious = []
    dists = []
    dists_seg = []
    model.eval()

    if not ((args.curObj== 'LPW') or (args.curObj=='Fuhl') or (args.curObj=='PupilNet')):
        opDict = {'id':[], 'img':[],
                  'scores':{'iou':[], 'pupil_c_error':[], 'iris_c_error':[]},
                  'pred':{'pupil_c':[], 'iris_c':[], 'mask':[]},
                  'gt':{'pupil_c':[],'iris_c':[],  'mask':[]}}        
    else:
        opDict = {'id':[], 'img':[],
                  'scores':{'pupil_c_error':[]},
                  'pred':{'pupil_c':[]},
                  'gt':{'pupil_c':[]}}  
      
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
            output, elOut, _, loss = out_tup
         
                    # Predicted centers
            pred_c_iri = elOut[:, 0:2].detach().cpu().numpy()
            pred_c_pup = elOut[:, 5:7].detach().cpu().numpy()
            
            predict = get_predictions(output)
            miou, iou, iou_sample = getSeg_metrics(labels.numpy(),
                                 predict.numpy(),
                                 cond[:, 1].numpy())#[1]
          
            # Center distance
            ptDist_iri = getPoint_metric(iris_center.numpy(),
                                         pred_c_iri,
                                         cond[:,1].numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points
            ptDist_pup = getPoint_metric(pupil_center.numpy(),
                                         pred_c_pup,
                                         cond[:,0].numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points
            # Angular distance
            angDist_iri = getAng_metric(elNorm[:, 0, 4].numpy(),
                                        elOut[:, 4].detach().cpu().numpy(),
                                        cond[:, 1].numpy())[0]
            angDist_pup = getAng_metric(elNorm[:, 1, 4].numpy(),
                                        elOut[:, 9].detach().cpu().numpy(),
                                        cond[:, 1].numpy())[0]
#            print (img.shape[0],pupil_center, pred_c_pup, ptDist_pup)            
            for i in range(0, img.shape[0]):
#                archNum = testObj.imList[imCounter, 1]
                if not ((args.curObj== 'LPW') or (args.curObj=='Fuhl') or (args.curObj=='PupilNet')):
                    opDict['pred']['mask'].append(predict[i,...].numpy().astype(np.uint8))
                    opDict['gt']['mask'].append(labels[i,...].numpy().astype(np.uint8))               
                    opDict['pred']['iris_c'].append(pred_c_iri[i,...])
                    opDict['gt']['iris_c'].append(iris_center[i,...])
                    opDict['scores']['iou'].append(iou_sample[i,...])
                    opDict['scores']['iris_c_error'].append(ptDist_iri)            
                    pred_img = predict[i].cpu().numpy()/3.0
                    label_img = labels[i].cpu().numpy()/3.0
                    inp = img[i].squeeze() * 0.5 + 0.5
                    img_orig = np.clip(inp,0,1)
                    img_orig = np.array(img_orig)
                    combine = np.hstack([img_orig,label_img,pred_img])
                    plt.imsave(path2op_mask+'/{}.jpg'.format(testObj.imList[imCounter, 0]),combine)
                    opDict['gt']['pupil_c'].append(pupil_center[i,...])
                    opDict['pred']['pupil_c'].append(pred_c_pup[i,...])
                opDict['id'].append(testObj.imList[imCounter, 0])
                opDict['scores']['pupil_c_error'].append(ptDist_pup)                
                opDict['img'].append(img[i,...].numpy())
                imCounter+=1
            
#%%
        print ('#########################################')
        print ('For paper')
        print (LOGDIR[25:])
        print ('Length of entires ', len(opDict))
        print ('#########################################')        
        if not ((args.curObj== 'LPW') or (args.curObj=='Fuhl') or (args.curObj=='PupilNet')):
            print ('Scores (IOU)         '+ '  Mean ' +'     ' +'std')
            print ('Overall mIoU         '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])),4))+'     '+ str(np.round(np.nanstd(np.array(opDict['scores']['iou'])),4)))
            print ('Background class     '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])[:,0]),4))+'     '+str(np.round(np.nanstd(np.array(opDict['scores']['iou'])[:,0]),4)))
            print ('Iris class           '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])[:,1]),4))+'     '+str(np.round(np.nanstd(np.array(opDict['scores']['iou'])[:,1]),4)))
            print ('Pupil class          '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])[:,2]),4))+'     '+ str(np.round(np.nanstd(np.array(opDict['scores']['iou'])[:,2]),4)))
            print ('#########################################')        
            print ('Scores (pixel error) '+ '  Mean ' +'     ' +'std')
            print ('Iris center error     '+ str(np.round(np.nanmean(np.array(opDict['scores']['iris_c_error'])),4))+'     '+str(np.round(np.nanstd(np.array(opDict['scores']['iris_c_error'])),4)))
        else:
            print ('Scores (pixel error) '+ '  Mean ' +'     ' +'std')
        print ('Pupil center error    '+ str(np.round(np.nanmean(np.array(opDict['scores']['pupil_c_error'])),4))+'     '+ str(np.round(np.nanstd(np.array(opDict['scores']['pupil_c_error'])),4)))

#%%
        logger = Logger(os.path.join(path2op,path_intermediate+'logs.log'))
        logger.write ('############################################')
        logger.write ('For paper')
        logger.write (LOGDIR[25:])
        logger.write ('Length of entires {}'.format(len(opDict)))
        logger.write ('#########################################')        
        if not ((args.curObj== 'LPW') or (args.curObj=='Fuhl') or (args.curObj=='PupilNet')):               
            logger.write ('Scores (IOU)         '+ '  Mean ' +'     ' +'std')
            logger.write ('Overall mIoU         '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])),4))+'     '+ str(np.round(np.nanstd(np.array(opDict['scores']['iou'])),4)))
            logger.write ('Background class     '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])[:,0]),4))+'     '+str(np.round(np.nanstd(np.array(opDict['scores']['iou'])[:,0]),4)))
            logger.write ('Iris class           '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])[:,1]),4))+'     '+str(np.round(np.nanstd(np.array(opDict['scores']['iou'])[:,1]),4)))
            logger.write ('Pupil class          '+ str(np.round(np.nanmean(np.array(opDict['scores']['iou'])[:,2]),4))+'     '+ str(np.round(np.nanstd(np.array(opDict['scores']['iou'])[:,2]),4)))
            logger.write ('#########################################')        
            logger.write ('Scores (pixel error) '+ '  Mean ' +'     ' +'std')
            logger.write ('Iris center error     '+ str(np.round(np.nanmean(np.array(opDict['scores']['iris_c_error'])),4))+'     '+str(np.round(np.nanstd(np.array(opDict['scores']['iris_c_error'])),4)))
        else:
            logger.write ('Scores (pixel error) '+ '  Mean ' +'     ' +'std')          
        logger.write ('Pupil center error    '+ str(np.round(np.nanmean(np.array(opDict['scores']['pupil_c_error'])),4))+'     '+ str(np.round(np.nanstd(np.array(opDict['scores']['pupil_c_error'])),4)))
        
   #%%     

        print('--- Saving output directory ---')
        f = open(os.path.join(path2op, path_intermediate+'opDict.pkl'), 'wb')
        pickle.dump(opDict, f)
        f.close()
        
#plt.figure(dpi=150)
#plt.hist(opDict['scores']['pupil_c_error'],bins=1000)
#plt.ylabel('Numer of samples')
#plt.xlabel('Error Quantity')
#plt.ylim([0,100])
#plt.axvline(np.nanmean(opDict['scores']['pupil_c_error']),color='r',label='mean')
#plt.title('LPW dataset')
#
#plt.axvline(np.nanmedian(opDict['scores']['pupil_c_error']),color='k',label='median')
#plt.legend()
