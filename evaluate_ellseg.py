#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 03:34:22 2021

@author: rakshit
"""

import os
import sys
import cv2
import copy
import torch
import argparse
import numpy as np

from loss import get_seg2ptLoss

from tqdm import tqdm
from pathlib import Path
from pprint import pprint

from utils import get_predictions
from modelSummary import model_dict
from helperfunctions import plot_segmap_ellpreds, getValidPoints
from helperfunctions import ransac, ElliFit, my_ellipse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', type=str, default='/media/rakshit/Monster/Datasets/Gaze-in-Wild',
                        help='path to eye videos')
    parser.add_argument('--save_maps', type=int, default=0,
                        help='save segmentation maps')
    parser.add_argument('--save_overlay', type=int, default=1,
                        help='save output overlay')
    parser.add_argument('--vid_ext', type=str, default='mp4',
                        help='process videos with given extension')
    parser.add_argument('--loadfile', type=str, default='./weights/all.git_ok',
                        help='choose the weights you want to evalute the videos with. Recommended: all')
    parser.add_argument('--align_width', type=int, default=1,
                        help='reshape videos by matching width, default: True')
    parser.add_argument('--eval_on_cpu', type=int, default=0,
                        help='evaluate using CPU instead of GPU')
    parser.add_argument('--check_for_string_in_fname', type=str, default='',
                        help='process video with a certain string in filename')
    parser.add_argument('--ellseg_ellipses', type=int, default=-1,
                        help='use ellseg proposed ellipses, if FALSE, it will fit an ellipse to segmentation mask')
    parser.add_argument('--skip_ransac', type=int, default=0,
                        help='if using ElliFit, it skips outlier removal')

    args = parser.parse_args()
    opt = vars(args)
    print('------')
    print('parsed arguments:')
    pprint(opt)
    return args

#%% Preprocessing functions and module
# Input frames must be resized to 320X240
def preprocess_frame(img, op_shape, align_width=True):
    if align_width:
        if op_shape[1] != img.shape[1]:
            sc = op_shape[1]/img.shape[1]
            width = int(img.shape[1] * sc)
            height = int(img.shape[0] * sc)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

            if op_shape[0] > img.shape[0]:
                # Vertically pad array
                pad_width = op_shape[0] - img.shape[0]
                if pad_width%2 == 0:
                    img = np.pad(img, ((pad_width//2, pad_width//2), (0, 0)))
                else:
                    img = np.pad(img, ((np.floor(pad_width/2), np.ceil(pad_width/2)), (0, 0)))
                scale_shift = (sc, pad_width)

            elif op_shape[0] < img.shape[0]:
                # Vertically chop array off
                pad_width = op_shape[0] - img.shape[0]
                if pad_width%2 == 0:
                    img = img[-pad_width/2:+pad_width/2, ...]
                else:
                    img = img[-np.floor(pad_width/2):+np.ceil(pad_width/2), ...]
                scale_shift = (sc, pad_width)

            else:
                scale_shift = (sc, 0)
        else:
            scale_shift = (1, 0)
    else:
        sys.exit('Height alignment not implemented! Exiting ...')

    img = (img - img.mean())/img.std()
    img = torch.from_numpy(img).unsqueeze(0).to(torch.float32) # Add a dummy color channel
    return img, scale_shift

#%% Forward operation on network
def evaluate_ellseg_on_image(frame, model):

    assert len(frame.shape) == 4, 'Frame must be [1,1,H,W]'

    with torch.no_grad():
        x4, x3, x2, x1, x = model.enc(frame)
        latent = torch.mean(x.flatten(start_dim=2), -1)
        elOut = model.elReg(x, 0)
        seg_out = model.dec(x4, x3, x2, x1, x)

    seg_out, elOut, latent = seg_out.cpu(), elOut.squeeze().cpu(), latent.squeeze().cpu()

    seg_map = get_predictions(seg_out).squeeze().numpy()

    ellipse_from_network = 1 if args.ellseg_ellipses == 1 else 0
    ellipse_from_output = 1 if args.ellseg_ellipses == 0 else 0
    no_ellipse = 1 if args.ellseg_ellipses == -1 else 0

    if ellipse_from_network:
        # Get EllSeg proposed ellipse predictions
        # Ellipse Centers -> derived from segmentation output
        # Ellipse axes and orientation -> Derived from latent space

        _, norm_pupil_center = get_seg2ptLoss(seg_out[:, 2, ...], torch.zeros(2, ), temperature=4)
        _, norm_iris_center  = get_seg2ptLoss(-seg_out[:, 0, ...], torch.zeros(2, ), temperature=4)

        norm_pupil_ellipse = torch.cat([norm_pupil_center, elOut[7:10]])
        norm_iris_ellipse  = torch.cat([norm_iris_center, elOut[2:5]])

        # Transformation function H
        _, _, H, W = frame.shape
        H = np.array([[W/2, 0, W/2], [0, H/2, H/2], [0, 0, 1]])

        pupil_ellipse = my_ellipse(norm_pupil_ellipse.numpy()).transform(H)[0][:-1]
        iris_ellipse  = my_ellipse(norm_iris_ellipse.numpy()).transform(H)[0][:-1]

    if ellipse_from_output:
        # Get ElliFit derived ellipse fits from segmentation mask

        seg_map_temp = copy.deepcopy(seg_map)
        seg_map_temp[seg_map_temp==2] += 1 # Pupil by PartSeg standard is 3
        seg_map_temp[seg_map_temp==1] += 1 # Iris by PartSeg standard is 2

        pupilPts, irisPts = getValidPoints(seg_map_temp, isPartSeg=False)

        if np.sum(seg_map_temp == 3) > 50 and type(pupilPts) is not list:
            if args.skip_ransac:
                model_pupil = ElliFit(**{'data': pupilPts})
            else:
                model_pupil = ransac(pupilPts, ElliFit, 15, 40, 5e-3, 15).loop()
        else:
            print('Not enough pupil points')
            model_pupil = type('model', (object, ), {})
            model_pupil.model = np.array([-1, -1, -1, -1, -1])

        if np.sum(seg_map_temp == 2) > 50 and type(irisPts) is not list:
            if args.skip_ransac:
                model_iris = ElliFit(**{'data': irisPts})
            else:
                model_iris = ransac(irisPts, ElliFit, 15, 40, 5e-3, 15).loop()
        else:
            print('Not enough iris points')
            model_iris = type('model', (object, ), {})
            model_iris.model = np.array([-1, -1, -1, -1, -1])
            model_iris.Phi = np.array([-1, -1, -1, -1, -1])
            # iris_fit_error = np.inf

        pupil_ellipse = np.array(model_pupil.model)
        iris_ellipse = np.array(model_iris.model)

    if no_ellipse:
        pupil_ellipse = np.array([-1, -1, -1, -1, -1])
        iris_ellipse = np.array([-1, -1, -1, -1, -1])

    return seg_map, latent.cpu().numpy(), pupil_ellipse, iris_ellipse

#%% Rescale operation to bring segmap, pupil and iris ellipses back to original res
def rescale_to_original(seg_map, pupil_ellipse, iris_ellipse, scale_shift, orig_shape):

    # Fix pupil ellipse
    pupil_ellipse[1] = pupil_ellipse[1] - np.floor(scale_shift[1]//2)
    pupil_ellipse[:-1] = pupil_ellipse[:-1]*(1/scale_shift[0])

    # Fix iris ellipse
    iris_ellipse[1] = iris_ellipse[1] - np.floor(scale_shift[1]//2)
    iris_ellipse[:-1] = iris_ellipse[:-1]*(1/scale_shift[0])

    if scale_shift[1] < 0:
        # Pad background
        seg_map = np.pad(seg_map, ((-scale_shift[1]//2, -scale_shift[1]//2), (0, 0)))
    elif scale_shift[1] > 0:
        # Remove extra pixels
        seg_map = seg_map[scale_shift[1]//2:-scale_shift[1]//2, ...]

    seg_map = cv2.resize(seg_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    return seg_map, pupil_ellipse, iris_ellipse

#%% Definition for processing per video
def evaluate_ellseg_per_video(path_vid, args, model):
    path_dir, full_file_name = os.path.split(path_vid)
    file_name = os.path.splitext(full_file_name)[0]

    if args.eval_on_cpu:
        device=torch.device("cpu")
    else:
        device=torch.device("cuda")

    if args.check_for_string_in_fname in file_name:
        print('Processing file: {}'.format(path_vid))
    else:
        print('Skipping video {}'.format(path_vid))
        return False

    vid_obj = cv2.VideoCapture(str(path_vid))

    FR_COUNT = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
    FR = vid_obj.get(cv2.CAP_PROP_FPS)
    H  = vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W  = vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH)

    path_vid_out = os.path.join(path_dir, file_name+'_ellseg.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(path_vid_out, fourcc, int(FR), (int(W), int(H)))

    # Dictionary to save output ellipses
    ellipse_out_dict = {}

    ret = True
    pbar = tqdm(total=FR_COUNT)

    counter = 0
    while ret:
        ret, frame = vid_obj.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.max() < 20:
            # Frame is quite dark, skip processing this frame.
            print('Skipping frame: {}'.format(counter))
            ellipse_out_dict[counter] = {'pupil': -1*np.ones(5, ), 'iris': -1*np.ones(5, )}
            continue

        frame_scaled_shifted, scale_shift = preprocess_frame(frame, (240, 320), args.align_width)

        input_tensor = frame_scaled_shifted.unsqueeze(0).to(device)

        # Run the prediction network
        seg_map, latent, pupil_ellipse, iris_ellipse = evaluate_ellseg_on_image(input_tensor, model)

        # Return ellipse predictions back to original dimensions
        seg_map, pupil_ellipse, iris_ellipse = rescale_to_original(seg_map,
                                                                   pupil_ellipse,
                                                                   iris_ellipse,
                                                                   scale_shift,
                                                                   frame.shape)

        # Generate visuals
        frame_overlayed_with_op = plot_segmap_ellpreds(frame, seg_map, pupil_ellipse, iris_ellipse)
        vid_out.write(frame_overlayed_with_op[..., ::-1])

        # Append to dictionary
        ellipse_out_dict[counter] = {'pupil': pupil_ellipse, 'iris': iris_ellipse}

        pbar.update(1)
        counter+=1

    vid_out.release()
    vid_obj.release()
    pbar.close()

    # Save out ellipse dictionary
    np.save(os.path.join(path_dir, file_name+'_pred.npy'), ellipse_out_dict)

    return True


if __name__=='__main__':
    args = parse_args()

    #%% Load network, weights and get ready to evalute
    netDict = torch.load(args.loadfile)

    model = model_dict['ritnet_v3']
    model.load_state_dict(netDict['state_dict'], strict=True)

    if not args.eval_on_cpu:
        model.cuda()

    #%% Read in each video
    path_obj = Path(args.path2data).rglob('*.mp4')

    for path_vid in path_obj:
        if '_ellseg' not in str(path_vid):
            evaluate_ellseg_per_video(path_vid, args, model)

