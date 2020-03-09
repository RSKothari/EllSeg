#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:05:17 2019

@author: aayush
"""
import cv2
import copy
import numpy as np

def augment(base, mask, pupil_c, elParam, choice=None):
    aug_base = np.zeros_like(base)
    aug_mask = np.zeros_like(mask)
    aug_pupil_c = copy.deepcopy(pupil_c)
    aug_pupilParam = copy.deepcopy(elParam[0])
    aug_irisParam = copy.deepcopy(elParam[1])
    
    badPup_c = True if np.all(pupil_c == -1) else False
    badPup = True if np.all(aug_pupilParam == -1) else False
    badIri = True if np.all(aug_irisParam == -1) else False
    
    index_value = np.random.randint(0, 8) if choice is None else choice
 
    if index_value == 0:
        # Flip left right
        aug_base = np.fliplr(base)
        aug_mask = np.fliplr(mask)
        
        aug_pupil_c[0] = base.shape[1] - aug_pupil_c[0] if not badPup_c else aug_pupil_c[0]
        aug_pupilParam[0] = base.shape[1] - elParam[0][0] if not badPup else aug_pupilParam[0]
        aug_pupilParam[-1] = 0.5*np.pi + elParam[0][-1] if not badPup else aug_pupilParam[-1]
        aug_irisParam[0] = base.shape[1] - elParam[1][0] if not badIri else aug_irisParam[0]
        aug_irisParam[-1] = 0.5*np.pi + elParam[1][-1] if not badIri else aug_irisParam[-1]

    elif index_value == 1:
        # Gaussian blur
        sigma_value=np.random.randint(2, 7)
        aug_base = cv2.GaussianBlur(base,(7,7),sigma_value)
        aug_mask = copy.deepcopy(mask)

    elif index_value == 2:
        # Gamma modification
        gamma = [0.6, 0.8, 1.2, 1.4][np.random.randint(0, 4)]
        table = 255.0*(np.linspace(0, 1, 256)**gamma)
        aug_base = cv2.LUT(base, table)
        aug_mask = copy.deepcopy(mask)

    elif index_value == 3:
        # Exposure +/- 25 pixel intensity
        aug_base = base.astype(np.float64) + (50*np.random.rand(1)-25)
        aug_base = np.clip(aug_base, 0, 255)
        aug_base = aug_base.astype(base.dtype)
        aug_mask = copy.deepcopy(mask)

    elif index_value == 4:
        # Gaussian noise
        #https://stackoverflow.com/questions/43699326/how-to-add-gaussian-noise-in-an-image-in-python-using-pymorph?rq=1
        mean = 0.0   # some constant
        std = 14*np.random.rand() + 2   # some constant (standard deviation)
        aug_base = base + np.random.normal(mean, std, base.shape)
        aug_base = np.clip(aug_base, 0, 255)  # we might get out of bounds due to noise
        aug_mask = copy.deepcopy(mask)

    elif index_value == 5:
        # Circular lines from pupil center
        yc, xc = (0.3 + 0.4*np.random.rand(1))*base.shape

        aug_base = copy.deepcopy(base)
        aug_mask = copy.deepcopy(mask)

        num_lines = np.random.randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*np.random.rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta)
            aug_base = cv2.line(aug_base, (x1, y1), (x2, y2), (255, 255, 255), 4)
        aug_base = aug_base.astype(np.uint8)
        '''
    elif index_value == 6:
        # Starburst pattern
        x=np.random.randint(1, 40)
        y=np.random.randint(1, 40)
        mode = np.random.randint(0, 2)
        starburst = cv2.imread('starburst_black.png', 0)
        aug_base = copy.deepcopy(base)
        if mode == 0:
            starburst = np.pad(starburst, pad_width=((0, 0), (x, 0)), mode='constant')
            starburst = starburst[:, :-x]
        if mode == 1:
            starburst = np.pad(starburst, pad_width=((0, 0), (0, x)), mode='constant')
            starburst = starburst[:, x:]

        aug_base[92+y:549+y, 0:400] = aug_base[92+y:549+y, 0:400]*(255 - starburst.astype(np.float))/255 + starburst
        aug_base = aug_base.astype(np.uint8)
        aug_mask = copy.deepcopy(mask)
        '''
    elif index_value == 6:
        # Rotate image
        ang = 30*2*(np.random.rand(1) - 0.5)
        center = (int(0.5*base.shape[1]), int(0.5*base.shape[0]))
        M = cv2.getRotationMatrix2D(center,
                                    ang,
                                    1.0)
        
        aug_base = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), flags=cv2.INTER_LANCZOS4)
        aug_mask = cv2.warpAffine(mask, M, (base.shape[1], base.shape[0]), flags=cv2.INTER_NEAREST)
        
        ang_rad = np.deg2rad(ang)
        R = np.array([[np.cos(ang_rad), -np.sin(ang_rad)],
                      [np.sin(ang_rad), np.cos(ang_rad)]]).squeeze()
        R = R.T
        aug_pupil_c = np.matmul(R, aug_pupil_c - np.array(center)) + np.array(center) # Center does rotate
        
        aug_pupilParam[:2] = np.matmul(R, aug_pupilParam[:2] - np.array(center)) + np.array(center)
        aug_pupilParam[-1] = aug_pupilParam[-1] - ang_rad if not badPup else aug_pupilParam[-1]
        aug_irisParam[:2] = np.matmul(R, aug_irisParam[:2] - np.array(center)) + np.array(center)
        aug_irisParam[-1] = aug_irisParam[-1] - ang_rad if not badIri else aug_irisParam[-1]
    
    elif index_value >=7:
        # Absolute no change
        aug_base = copy.deepcopy(base)
        aug_mask = copy.deepcopy(mask)

    return (aug_base.astype(np.uint8),
            aug_mask.astype(np.int),
            aug_pupil_c,
            (aug_pupilParam, aug_irisParam))

def getRandomLine(xc, yc, theta):
    x1 = xc - 50*np.random.rand(1)*(1 if np.random.rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150*np.random.rand(1) + 50)*(1 if np.random.rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    return x1, y1, x2, y2
	
def normalizer(image):
    return np.uint8((image-image.min())*255/(image.max()-image.min()))

'''
UNSED CODES
def transform_aug(base, mask,randomize):
    base1=base[:,:,0]
    if randomize:
        index_value= np.random.randint(0,20)
    else:
        index_value=20
    #print (index_value)
#    index_value=0
    ##flips the image and the mask
    if index_value==0:
        aug_base=cv2.flip(base1,-1)
        aug_mask=cv2.flip(mask,-1)
#        base1=base[:,:,0]
        #aug_base=torch.flip(torch.from_numpy(base1),dims=(1,)).numpy()
       # aug_base=np.reshape(base1,base1.shape+(1,))
        #aug_mask=torch.flip(torch.from_numpy(mask),dims=(1,)).numpy()

    ##contrast limited histogram equalization
    elif index_value==1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))##try (4,4) orig (8,8)
        temp= clahe.apply(base1)
        aug_base =(temp)
        aug_mask=mask

    ##averaging filter Random value 3,5,7,9,11
    elif index_value==2:
#        base1=base[:,:,0]
        kernel_width=3+2*np.random.randint(0,5)
        kernel = np.ones((kernel_width,kernel_width),np.float32)/kernel_width/kernel_width
        aug_base = cv2.filter2D(base1,-1,kernel)
        aug_mask=mask

    ##Gaussian Blur sigma_value=0.1 to 1 in mulples of 0.1 (Random) 7X7
    elif index_value==3:
        sigma_value=np.random.randint(0,10)/10
        aug_base= cv2.GaussianBlur(base1,(7,7),sigma_value)
        aug_mask=mask

    ##local centering an image index_valonly one channel
    elif index_value==4:
        aug_base=normalizer(base1-base1.mean())
        aug_mask=mask

    ##Adding Gaussian noise of 0 mean and 2 STD
    elif index_value==5:
        #https://stackoverflow.com/questions/43699326/how-to-add-gaussian-noise-in-an-image-in-python-using-pymorph?rq=1
        mean = 0.0   # some constant
        std = 2.0    # some constant (standard deviation)
        noisy_img = base1+ np.random.normal(mean, std, base1.shape)
        aug_base = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise
        aug_mask=mask

    ###Downsample by factor of 2 to 5 and upsample
    elif index_value==6:
        factor=np.random.randint(2,6)
        aug_base=np.array(Image.fromarray(base1).resize((int(base1.shape[1]/factor),int(base1.shape[0]/factor)), Image.ANTIALIAS).resize((int(base.shape[1]),int(base.shape[0])),Image.CUBIC))
        aug_mask=np.array(Image.fromarray(mask).resize((int(base1.shape[1]/factor),int(base1.shape[0]/factor)), Image.ANTIALIAS).resize((int(base.shape[1]),int(base.shape[0])),Image.CUBIC))

    ###Translation by (2-10 pixels randomly)
    elif index_value==7:
        factor=2*np.random.randint(1,6)
        aug_base=np.roll(base1,factor)
        aug_mask=np.roll(mask,factor)

    ##rotation from 0 angree to 30 angree
    elif index_value==8:
        rows,cols = base1.shape
        theta=np.random.randint(0,30)
        aug_base=np.array(Image.fromarray(base1).rotate(theta))
        aug_mask=np.array(Image.fromarray(mask).rotate(theta))
#        center_y=int(rows/2)
#        center_x=int(cols/2)
#        alpha= np.cos(theta*np.pi/180)
#        beta= np.sin(theta*np.pi/180)
#        M=np.array([[alpha, beta, (1-alpha)*center_x-beta*center_y],[-beta,alpha, (1-alpha)*center_y+beta*center_x]])
#        aug_base = cv2.warpAffine(base1,M,(cols,rows))
#        aug_mask = cv2.warpAffine(mask,M,(cols,rows))

    ##Frensel patterns random size
    elif index_value==9:
        circular_rings=np.zeros((base1.shape))
        rows,cols = base1.shape
        factor=5*np.random.randint(1,4)
        for i in range (int(150/factor)):
            cv2.circle((circular_rings),(int(rows/2),int(cols/2)),i*factor,(24,24,24),1)
        aug_base = np.uint8((base1+circular_rings))
        aug_mask=mask


    ###Moire patches straight line
    elif index_value==10:
        straight=np.zeros((base1.shape))
        for i in range (int(30)):
            cv2.line(straight,(0,np.random.randint(0,base1.shape[1])),(base1.shape[0],np.random.randint(0,base1.shape[1])),(24,24,24),1)
            cv2.line(straight,(np.random.randint(0,base1.shape[0]),0),(np.random.randint(0,base1.shape[0]),base1.shape[1]),(24,24,24),1)
        aug_base =np.uint8(straight+base1)
        aug_mask=mask

    ###Moire patterns circular lines
    elif index_value==11:
        circular_rings=np.zeros((base1.shape))
        rows,cols = base1.shape
        factor=5*np.random.randint(1,4)
        for i in range (int(150/factor)):
            cv2.circle((circular_rings),(int(rows/2),int(cols/2)),i*factor,(24,24,24),1)
            cv2.circle((circular_rings),(int(rows/2)+factor*3,-factor*4+int(cols/2)),int(i*factor*1.2),(24,24,24),1)
            cv2.circle((circular_rings),(int(rows/2)-factor*8,factor*5+int(cols/2)),int(i*factor/0.6),(24,24,24),1)
        aug_base = np.uint8((base1+circular_rings))
        aug_mask=mask

        #crop ans resize
    elif index_value==12:
        base2=np.zeros((int(base1.shape[0]/2),int(base1.shape[1]/2)))
        base2=base1[int(base.shape[0]/4):int(base.shape[0]*3/4),int(base.shape[1]/4):int(base.shape[1]*3/4)]
        aug_base=np.array(Image.fromarray(base2).resize((int(base.shape[1]),int(base.shape[0])),Image.CUBIC))
        base2=np.zeros((int(base1.shape[0]/2),int(base1.shape[1]/2)))
        base2=mask[int(base.shape[0]/4):int(base.shape[0]*3/4),int(base.shape[1]/4):int(base.shape[1]*3/4)]
        aug_mask=np.array(Image.fromarray(base2).resize((int(base.shape[1]),int(base.shape[0])),Image.CUBIC))

   ##return without changing
    else:
        aug_base=base[:,:,0]
        aug_mask=mask

    aug_base=np.reshape(aug_base,aug_base.shape+(1,))
    return aug_base, aug_mask
'''