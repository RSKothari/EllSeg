#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:28:20 2020

@author: rakshit

Input options:
    disentagle: 0, 1, 2 [None, Zisserman, GradientReversal]
    overfit: 0 [None, int > 0 is the # of batches you want to experiment]
    selfCorr: 0, 1 [None, self consistency loss active]
    disp: 0, 1 [Display output during training to visualize process]
"""

from pprint import pprint
import argparse
import torch

def parse_precision(prec):
    if prec==32:
        return torch.float32
    elif prec==64:
        return torch.float32
    elif prec==16:
        return torch.float16
    else:
        print('Invalid precision. Reverting to float32.')
        return torch.float32

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--curObj', type=str, help='select curriculum to train on', required=True)
    parser.add_argument('--path2data', type=str, default='/media/rakshit/tank', help='path to dataset')
    parser.add_argument('--epochs', type=int, default=40, help='total number of epochs')
    parser.add_argument('--model', type=str, default='ritnet_v1', help='select model')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=12, help='select a batchsize')
    parser.add_argument('--resume', type=int, default=0, help='resume?')
    parser.add_argument('--loadfile', type=str, default='', help='load experiment')
    parser.add_argument('--expname', type=str, default='dev', help='experiment number')
    parser.add_argument('--prec', type=int, default=32, help='precision. 16, 32, 64')
    parser.add_argument('--disp', type=int, default=0, help='display intermediate ouput')
    parser.add_argument('--workers', type=int, default=6, help='number of workers')
    parser.add_argument('--overfit', type=int, default=0, help='overfit to N batches?')
    parser.add_argument('--selfCorr', type=int, default=0, help='self regulation?')
    parser.add_argument('--test_mode', type=str, default='baseline', help='testing strategy?')
    parser.add_argument('--disentangle', type=int, default=0, help='Explicit dataset bias removal?')

    args = parser.parse_args()
    opt = vars(args)
    print('------')
    print('parsed arguments:')
    pprint(opt)
    args.prec = parse_precision(args.prec)
    return args

if __name__ == '__main__':
    opt = parse_args()
