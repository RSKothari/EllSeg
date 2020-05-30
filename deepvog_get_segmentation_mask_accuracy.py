from DeepVOG_model import load_DeepVOG
import skimage.io as ski
import numpy as np
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from helperfunctions import mypause
from torch.utils.data import DataLoader
from utils import generateImageGrid, points_to_heatmap
from CurriculumLib import readArchives, listDatasets, generate_fileList
from CurriculumLib import selDataset, selSubset, DataLoader_riteyes
import subprocess

def normalize_image(img1):
    return (img1-np.min(img1))/(np.max(img1)-np.min(img1))
  
def get_overlap(pred_bbox,gt_bbox):
    pred_i = pred_bbox == True
    label_i = gt_bbox == True
    intersection = np.logical_and(label_i, pred_i)
    union = np.logical_or(label_i, pred_i)
    iou_score = np.sum(intersection.numpy())/np.sum(union.numpy())
    return iou_score
  
if __name__=='__main__':
    path2data = '/media/aaa/hdd/ALL_model/giw_e2e/Dataset'
    path2h5 = os.path.join(path2data, 'All')
    path2arc_keys = os.path.join(path2data, 'MasterKey')
    '''
    AllDS = readArchives(path2arc_keys)
    datasets_present, subsets_present = listDatasets(AllDS)
    print('Datasets present -----')
    print(datasets_present)
    print('Subsets present -----')
    print(subsets_present)

    nv_subs1 = ['nvgaze_female_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 5) for j in range(0, 3)]
    nv_subs2 = ['nvgaze_male_{:02}_public_50K_{}'.format(i+1, j+1) for i in range(0, 5) for j in range(0, 3)]
    lpw_subs = ['LPW_{}'.format(i+1) for i in range(0, 12)]
    subsets = nv_subs1 + nv_subs2 + lpw_subs + ['none', 'train']

    AllDS = selDataset(AllDS, ['OpenEDS', 'UnityEyes', 'NVGaze', 'LPW', 'riteyes_general'])
    AllDS = selSubset(AllDS, subsets)
    dataDiv_obj = generate_fileList(AllDS, mode='vanilla', notest=True)
    trainObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'train', True, (480, 640), 0.5)
    validObj = DataLoader_riteyes(dataDiv_obj, path2h5, 0, 'valid', False, (480, 640), 0.5)
    '''
    
    cond_name='riteyes_general'
    f = os.path.join('curObjects','baseline', 'cond_'+cond_name+'.pkl')
    trainObj, validObj, testObj = pickle.load(open(f, 'rb'))
    trainObj.path2data = path2h5
    testObj.path2data = path2h5
    validObj.path2data = path2h5


    trainLoader = DataLoader(testObj,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             drop_last=True)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    totTime = []
    startTime = time.time()
    opDict = {'Expected_filename':[], 'pupil_center_x':[],'pupil_center_y':[],\
              'counter_map':[],'average_mIoU':[]}
    Expected_filename=[]
    pupil_center_x=[]
    pupil_center_y=[]
    counter=0
    model = load_DeepVOG()
    average_mIoU=[]
    counter=1
    counter_map=[]
    save_no=0
    for bt, data in enumerate(trainLoader):
        print (bt)
        I, mask, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = data
        for i in range (I.shape[0]):
            print (imInfo)
            img=normalize_image(I[i].numpy())
            img_3=np.zeros((1,240,320,3))
            img_3[:,:,:,0]=img
            img_3[:,:,:,1]=img
            img_3[:,:,:,2]=img
            prediction = model.predict(img_3)
            gt_mask=mask[i]==2
            pred_mask=prediction[0,:,:,1]>0.5
            img=Image.fromarray(np.uint8(img_3[0]*255))
            average_mIoU.append(get_overlap(pred_mask,gt_mask))
            img.save('/media/aaa/hdd/ALL_model/giw_e2e/Dataset/testdataset/'+cond_name+'_deepvog/'+str(counter)+'.png')
            counter_map.append(counter)
            pupil_center_x.append(pupil_center[i][0])
            pupil_center_y.append(pupil_center[i][1])
            Expected_filename.append(imInfo[i,0])          
            counter=counter+1
            
        if len (counter_map)>4000:
            opDict['Expected_filename']=(Expected_filename)
            opDict['pupil_center_x']=(pupil_center_x)
            opDict['pupil_center_y']=(pupil_center_y)
            opDict['counter_map']=(counter_map)
            opDict['average_mIoU']=(average_mIoU)
        #%%
            filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
            print('--- Saving output directory ---')
            f = open(filename, 'wb')
            pickle.dump(opDict, f)
            f.close()

            opDict = {'Expected_filename':[], 'pupil_center_x':[],'pupil_center_y':[],\
                      'counter_map':[],'average_mIoU':[]}
            Expected_filename=[]
            pupil_center_x=[]
            pupil_center_y=[]
            average_mIoU=[]
            counter_map=[]
            save_no+=1
    opDict['Expected_filename']=(Expected_filename)
    opDict['pupil_center_x']=(pupil_center_x)
    opDict['pupil_center_y']=(pupil_center_y)
    opDict['counter_map']=(counter_map)
    opDict['average_mIoU']=(average_mIoU)
#%%
    filename='/media/aaa/hdd/ALL_model/giw_e2e/op/DeepVOG_GT_Pupil_center_and_mIoU'+cond_name+str(save_no)+'.pkl'              
    print('--- Saving output directory ---')
    f = open(filename, 'wb')
    pickle.dump(opDict, f)
    f.close()
    
          #saving Images
#            print (i)
#            img=Image.fromarray(I[i].numpy())
            
#            mask_current=mask[i].numpy()
#            image_current=np.uint8(I[i].numpy())
#            mask_current[mask_current>1]=1
#            image_current=image_current*mask_current
#            image_current[np.where(mask_current==np.int32(0))]=127
#
#            img=Image.fromarray(np.uint8(image_current))

#            img.save('/media/aaa/hdd/ALL_model/giw_e2e/Dataset/testdataset/riteyes_general/'+str(int(imInfo[i,0]))+\
#                     '_'+str(int(pupil_center[i][0]))+'_'+str(int(pupil_center[i][1]))+'.png')
            
          
#          Ssaving GT
#            Expected_filename.append(str(int(imInfo[i,0]))+\
#                     '_'+str(int(pupil_center[i][0]))+'_'+str(int(pupil_center[i][1])))
#            pupil_center_x.append(pupil_center[i][0].numpy())
#            pupil_center_y.append(pupil_center[i][1].numpy())
#            
#            if len(pupil_center_x)==6000:
#                print ('here')
#                opDict['Expected_filename']=(Expected_filename)
#                opDict['pupil_center_x']=(pupil_center_x)
#                opDict['pupil_center_y']=(pupil_center_y)
#                Expected_filename=[]
#                pupil_center_x=[]
#                pupil_center_y=[]
#                
#                filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/GT_Pupil_center_'+cond_name+str(counter)+'.pkl'              
#                print('--- Saving output directory ---')
#                f = open(filename, 'wb')
#                pickle.dump(opDict, f)
#                f.close()
#                counter+=1
#                opDict = {'Expected_filename':[], 'pupil_center_x':[],'pupil_center_y':[]}                
#    print ('here')
#    opDict['Expected_filename']=(Expected_filename)
#    opDict['pupil_center_x']=(pupil_center_x)
#    opDict['pupil_center_y']=(pupil_center_y)
#    Expected_filename=[]
#    pupil_center_x=[]
#    pupil_center_y=[]
    
#    filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/GT_Pupil_center_'+cond_name+str(counter)+'.pkl'              
#    print('--- Saving output directory ---')
#    f = open(filename, 'wb')
#    pickle.dump(opDict, f)
#    f.close()
#    counter+=1
#    opDict = {'Expected_filename':[], 'pupil_center_x':[],'pupil_center_y':[]}                
# 
#    print('Time for 1 epoch: {}'.format(np.sum(totTime)))

#%%
#def test_if_model_work(model, image):
#    model = load_DeepVOG()
#    img = np.zeros((1, 240, 320, 3))
#    img[:,:,:,:] = (ski.imread("test_image.png")/255).reshape(1, 240, 320, 1)
#    prediction = model.predict(img)
#    ski.imsave("test_prediction.png", prediction[0,:,:,1])
#
#if __name__ == "__main__":
#    # If model works, the "test_prediction.png" should show the segmented area of pupil from "test_image.png"
#    test_if_model_work()
