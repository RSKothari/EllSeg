#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:32:15 2020

@author: rakshit
"""

from models.RITnet_v1 import DenseNet2D as DN_v1
from models.RITnet_v2 import DenseNet2D as DN_v2
from models.RITnet_v3 import DenseNet2D as DN_v3
from models.RITnet_v4 import DenseNet2D as DN_v4
from models.RITnet_v5 import DenseNet2D as DN_v5
from models.RITnet_v6 import DenseNet2D as DN_v6
from models.RITnet_v7 import DenseNet2D as DN_v7
from models.deepvog_pytorch import DeepVOG_pytorch

model_dict = {}
model_dict['ritnet_v1'] = DN_v1()
model_dict['ritnet_v2'] = DN_v2()
model_dict['ritnet_v3'] = DN_v3()
model_dict['ritnet_v4'] = DN_v4()
model_dict['ritnet_v5'] = DN_v5()
model_dict['ritnet_v6'] = DN_v6()
model_dict['ritnet_v7'] = DN_v7()
model_dict['deepvog'] = DeepVOG_pytorch()
