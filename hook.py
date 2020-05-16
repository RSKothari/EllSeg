#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:09:33 2020

@author: aaa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections

# dummy data: 10 batches of images with batch size 16
dataset = [torch.rand(2,3,224,224) for _ in range(10)]

# network: a resnet50
#net = tmodels.resnet50(pretrained=True)

# a dictionary that keeps saving the activations as they come
activations = collections.defaultdict(list)
def save_activation(name, mod, inp, out):
	activations[name].append(out.cpu())

# Registering hooks for all the Conv2d layers
# Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
# called repeatedly at different stages of the forward pass (like RELUs), this will save different
# activations. Editing the forward pass code to save activations is the way to go for these cases.
for name, m in model.named_modules():
	if type(m)==nn.Conv2d:
		# partial to assign the layer name to each hook
		m.register_forward_hook(partial(save_activation, name))

# forward pass through the full dataset
#for batch in dataset:
#	out = model(batch)

# concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

# just print out the sizes of the saved activations as a sanity check
for k,v in activations.items():
	print (k, v.size())