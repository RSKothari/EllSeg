# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:36:48 2020

@author: Rudra
"""

import pickle as pkl

# NVGaze
nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 4) for j in range(0, 4)]
nv_subs_train = nv_subs1 + nv_subs2

nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 4)]
nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(5, j+1) for j in range(0, 4)]
nv_subs_test = nv_subs1 + nv_subs2

# OpenEDS
openeds_train = ['train']
openeds_test = ['validation']

# LPW
lpw_subs_train = ['LPW_{}'.format(i+1) for i in range(0, 16)]
lpw_subs_test = ['LPW_{}'.format(i+1) for i in range(16, 22)]

# S-General
riteyes_subs_train = ['riteyes_general_{}'.format(i+1) for i in range(0, 18)]
riteyes_subs_test = ['riteyes_general_{}'.format(i+1) for i in range(18, 24)]

# Fuhl
fuhl_subs_train =  ['data set I', 'data set II', 'data set III', 'data set IV',
                    'data set IX', 'data set V', 'data set VI', 'data set VII',
                    'data set VIII', 'data set X', 'data set XI', 'data set XII',
                    'data set XIII', 'data set XIV', 'data set XIX', 'data set XVI',
                    'data set XVII', 'data set XVIII', 'data set XX', 'data set XXI',
                    'data set XXII', 'data set XXIII', 'data set XV', 'data set XXIV'][::2]
fuhl_subs_test =  ['data set I', 'data set II', 'data set III', 'data set IV',
                    'data set IX', 'data set V', 'data set VI', 'data set VII',
                    'data set VIII', 'data set X', 'data set XI', 'data set XII',
                    'data set XIII', 'data set XIV', 'data set XIX', 'data set XVI',
                    'data set XVII', 'data set XVIII', 'data set XX', 'data set XXI',
                    'data set XXII', 'data set XXIII', 'data set XV', 'data set XXIV'][1::2]

# PupilNet
PN_subs_train = ['data set new I', 'data set new II', 'data set new III', 'data set new IV', 'data set new V'][::2]
PN_subs_test = ['data set new I', 'data set new II', 'data set new III', 'data set new IV', 'data set new V'][1::2]

DS_train = {'NVGaze':nv_subs_train,
            'OpenEDS':openeds_train,
            'LPW':lpw_subs_train,
            'Fuhl':fuhl_subs_train,
            'PupilNet': PN_subs_train,
            'riteyes_general': riteyes_subs_train}

DS_test = {'NVGaze':nv_subs_test,
            'OpenEDS':openeds_test,
            'LPW':lpw_subs_test,
            'Fuhl':fuhl_subs_test,
            'PupilNet': PN_subs_test,
            'riteyes_general': riteyes_subs_test}

DS_selections = {'train': DS_train,
                 'test' : DS_test}

pkl.dump(DS_selections, open('dataset_selections.pkl', 'wb'))
