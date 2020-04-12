# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:01:12 2020

@author: Rudra
"""

import torch
import torch.nn.functional as F

x1 = torch.tensor([10, 0, 0, 0, 0]).to(torch.float)
x2 = torch.tensor([2, 2, 2, 2, 2]).to(torch.float)
y = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5]).to(torch.float)

KL_loss = F.kl_div(F.log_softmax(x2, dim=0), y)
logSoft_loss = -F.log_softmax(x2).mean()
print('KL loss: {}'.format(KL_loss))
print('Log soft loss: {}'.format(logSoft_loss))