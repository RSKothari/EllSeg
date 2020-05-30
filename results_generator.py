#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:28:57 2020

@author: aayush
"""

import pickle
import scipy.signal as sig
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns 
import pylab

params = {'legend.fontsize': 30,
          'figure.figsize': (10,5),
         'axes.labelsize': 30,
         'axes.titlesize':30,
         'xtick.labelsize':30,
         'ytick.labelsize':30}
pylab.rcParams.update(params)  

dataset_names=['Fuhl','LPW','NVGaze','riteyes_general','OpenEDS','PupilNet']
dataset_names=['OpenEDS','NVGaze','riteyes_general']
#model_names=['ritnet_v5','ritnet_v4','ritnet_v6','ritnet_v2','ritnet_v1']#,'ritnet_v2','ritnet_v3']#,'deepvog']
#model_names=['ritnet_v6','ritnet_v2']#,'ritnet_v6','ritnet_v2','ritnet_v1']#,'ritnet_v2','ritnet_v3']#,'deepvog']
model_names=['ritnet_v5','ritnet_v6','ritnet_v4','ritnet_v2']#,'ritnet_v6','ritnet_v2','ritnet_v1']#,'ritnet_v2','ritnet_v3']#,'deepvog']
#v5, v6 , v1 , v4, v2, v3
test_cases=['_0_0']#,'_1_0']
include_deepvog=False
include_ellipse_fit_data=False
include_Excuse=False

figures_to_plot='part_3_2' ##see description below

#part_3_1  :Detection Rate Vs Pixel Error: 3x2 pupil and iris center estimates of all 3 models 
#part_3_2  :For OpenEDS dataset only..change that 
#           plot how good is ellipse fit mIoU
#part_3_3  :Show all performance on semantic segmentation
 

if figures_to_plot=='part_3_3': 
    model_names=['ritnet_v5','ritnet_v4','ritnet_v6','ritnet_v2']#,'ritnet_v6','ritnet_v2','ritnet_v1']#,'ritnet_v2','ritnet_v3']#,'deepvog']
    include_deepvog=True

if figures_to_plot=='part_3_1': 
    model_names=['ritnet_v6','ritnet_v2']#,'ritnet_v6','ritnet_v2','ritnet_v1']#,'ritnet_v2','ritnet_v3']#,'deepvog']
    include_ellipse_fit_data=True
    include_Excuse=True

if figures_to_plot=='part_2_1':
    model_names=['ritnet_v4','deepvog','ritnet_v5']#,'deepvog']
    include_deepvog=True
 
if figures_to_plot=='part_3_2': 
    model_names=['ritnet_v5','ritnet_v6','ritnet_v4','ritnet_v2']#,'ritnet_v6','ritnet_v2','ritnet_v1']#,'ritnet_v2','ritnet_v3']#,'deepvog']
    include_deepvog=False
    include_ellipse_fit_data=False
    include_Excuse=False

    
params = {'legend.fontsize': 40,
          'figure.figsize': (20,10),
         'axes.labelsize': 40,
         'axes.titlesize':40,
         'xtick.labelsize':40,
         'ytick.labelsize':40}
pylab.rcParams.update(params)                     
#if figures_to_plot=='part_3_2': 
#    params = {'legend.fontsize': 60,
#              'figure.figsize': (20,10),
#             'axes.labelsize': 60,
#             'axes.titlesize':60,
#             'xtick.labelsize':60,
#             'ytick.labelsize':60}
#    pylab.rcParams.update(params) 

def save_plot_hist(data_to_plot,ylabels,lower_lim,upper_lim, mean_values,title,figname,dataset_name,gs,m,n,figures_to_plot):
    data_listx=[]
    data_listy=[]
    if (figures_to_plot=='part_3_1') or (figures_to_plot=='part_2_1') :
        ax = plt.subplot(gs[n, m])
    else:
        ax = plt.subplot(gs[m, n])      
    print (m,n)
#    if figures_to_plot=='part_2_1':
    lst =data_to_plot
    pad = len(max(lst, key=len))
    data_to_plot=([(list(i)+[100]*(pad-len(i))) for i in lst])
    if figures_to_plot=='part_2_1':

        for i in range(len(ylabels)):
            data_listx+=list(data_to_plot[i])
            data_listy+=list([ylabels[i]]*len(data_to_plot[i]))
            if (ylabels[i][-5:])=='EpSeg':
                plt.hist(np.array(data_to_plot[i]),bins=100000, range=(0,100),linestyle='--',linewidth=3,density='True',histtype='step', cumulative=100,label=ylabels[i])
            elif (ylabels[i][-5:])=='ained':
                plt.hist(np.array(data_to_plot[i]),bins=100000, range=(0,100),linestyle=':',linewidth=3,density='True',histtype='step', cumulative=100,label=ylabels[i])
            else:
                plt.hist(np.array(data_to_plot[i]),bins=100000, range=(0,100),linewidth=3,density='True',histtype='step', cumulative=100,label=ylabels[i])

    if figures_to_plot=='part_3_1':

        for i in range(len(ylabels)):
            data_listx+=list(data_to_plot[i])
            data_listy+=list([ylabels[i]]*len(data_to_plot[i]))
            if (ylabels[i][-3:])=='w/L':
                plt.hist(np.array(data_to_plot[i]),bins=100000, range=(0,100),linewidth=3,density='True',histtype='step', cumulative=100,label=ylabels[i])
            elif (ylabels[i][-3:])=='/Ef':
                plt.hist(np.array(data_to_plot[i]),bins=100000, range=(0,100),linestyle='--',linewidth=3,density='True',histtype='step', cumulative=100,label=ylabels[i])
            else:
                plt.hist(np.array(data_to_plot[i]),bins=100000, range=(0,100),linewidth=3,density='True',histtype='step', cumulative=100,label=ylabels[i])

    ax.set_xlim([lower_lim,upper_lim])
#    ax.set_xlim([0.7,3])
#    ax.set_ylim([0.5,1.0])

    if (figures_to_plot=='part_3_1') or (figures_to_plot=='part_2_1') :
        if (n==1) & (m==0):
            ax.set_ylabel('Iris Detection Rate')
        if m>0:
            ax.set_yticks([])
        if (n==1) & (m==1):
            ax.text(2, -.20, 'Pixel Error', ha='center',fontsize=pylab.rcParams['axes.labelsize'])
        if (n==0):
            ax.set_xticks([])
            if (m==0):
                ax.set_title('OpenEDS')
                ax.set_ylabel('Pupil Detection Rate')
#                ax.legend(loc='lower right', ncol=2)
#                ax.legend(loc='upper center',framealpha=1,bbox_to_anchor=(0, 1.4),ncol=3)

            if (m==1):
                ax.set_title('NVGaze')
#                
#                fig.legend(loc='lower left', bbox_to_anchor= (0, 1), ncol=4,
#            borderaxespad=0, frameon=False)

#                ax.legend(loc='upper center', ncol=3)
#                ax.legend(loc='upper center',framealpha=1,bbox_to_anchor=(0, 1.4),ncol=3)
#                ax.legend(bbox_to_anchor=(0, 1),ncol=6)
#                gs.tight_layout(fig)
            if (m==2):
                ax.set_title('RITeyes-general') 
#    fig.legend(loc=2, mode='expand', numpoints=1, ncol=1, fancybox = False,
#         fontsize=40, labels=ylabels)#,bbox_to_anchor=(0.5, 0.5))
#    if figures_to_plot=='part_2_1':
#        if 
#        ax.legend(loc='lower right', ncol=2)

def save_plot(data_to_plot,ylabels,lower_lim,upper_lim, mean_values,title,figname,dataset_name,gs,m,n):
    data_listx=[]
    data_listy=[]
    ax = plt.subplot(gs[m, n])
    print (m,n)
    for i in range(len(ylabels)):
        data_listx+=list(data_to_plot[i])
        data_listy+=list([ylabels[i]]*len(data_to_plot[i]))
    
    datDf=pd.DataFrame({dataset_name:data_listy})
    datDf['']=data_listx
    ax = sns.boxplot(x='',y=dataset_name, data=datDf, whis=np.inf,palette="Set3")
    if mean_values is not None:    
        datDf2=pd.DataFrame({dataset_name:list(ylabels)})
        datDf2['']=list(mean_values)
        ax = sns.scatterplot(x='',y=dataset_name, data=datDf2,s=100,marker='^',color= sns.dark_palette("purple"))#sns.xkcd_rgb["pale red"])#palette="Set1",color="Red",marker='+')  
    ax.set_xlim([lower_lim,upper_lim])

#    if figures_to_plot=='part_3_2':
#        if n>0:
#            ax.set_yticks([])
#        if n==1:
#            print ('here')
#            ax.text(1, 7, 'Bounding box overlap IoU', ha='center',fontsize=pylab.rcParams['axes.labelsize'])
#        if n==3:
#            ax.text(1, 7, 'Orientation difference [\u00b0]', ha='center',fontsize=pylab.rcParams['axes.labelsize'])
#        ax.set_title(title)
#        if n>0:
#            ax.set_ylabel('')
    if figures_to_plot=='part_3_2':
        if (m==0):
            ax.set_title(title)
        if not (m==2):
            ax.set_xticks([])
        if n>0:
            ax.set_yticks([])

        if (n==1) & (m==2):
#            print ('here')
            ax.text(1, 4.5, 'Bounding box overlap IoU', ha='center',fontsize=pylab.rcParams['axes.labelsize'])

        if (n==3) & (m==2):
#            print ('here')
            ax.text(1, 4.5, 'Orientation difference [\u00b0]', ha='center',fontsize=pylab.rcParams['axes.labelsize'])
        if n>0:
            ax.set_ylabel('')
            
    elif figures_to_plot=='part_3_3':
        if (m==0):
            ax.set_title(title)
        if not (m==2):
            ax.set_xticks([])
        if n>0:
            ax.set_yticks([])
        if (n==2) & (m==2):
            print ('here')
            ax.text(1.2, 5.5, 'IoU scores', ha='center',fontsize=pylab.rcParams['axes.labelsize'])

        if n>0:
            ax.set_ylabel('')

    else:
        if (m==0):
            ax.set_title(title)
        if not (m==2):
            ax.set_xticks([])
        if n>0:
            ax.set_yticks([])
        
    

counter=0
fig = plt.figure(figsize=(60,20))
gs=gridspec.GridSpec(3, 6)
gs.update(left=0.05, right=0.95, wspace=0.05)

if 'part_3' in figures_to_plot:
    for dataset_name in dataset_names:
        data_pupil_c_error_un=[]
        data_pupil_c_error=[]
        data_iris_c_error_un=[]
        data_iris_c_error=[]
        data_pupil_iou=[]
        data_iris_iou=[]
        data_overall_iou=[]
        data_background_iou=[]
        data_iris_el_iou=[]
        data_pupil_el_iou=[]
        data_iris_el_angular_error=[]
        data_pupil_el_angular_error=[]
        
        save_filename=[]
        for model_name in model_names:
            for test_case in test_cases:
                filename='/media/aaa/hdd/ALL_model/giw_e2e/op/'+dataset_name+'/'+\
                        model_name+'/'+test_case+'opDict_scores.pkl'
                f = open(filename, 'rb')
                data=pickle.load(f)
                f.close()
                print (model_name, test_case)
                filename='/media/aaa/hdd/ALL_model/giw_e2e/op/'+dataset_name+'/'+\
                        model_name+'/'+test_case+'el_opDict.pkl'
                f = open(filename, 'rb')
                data_el=pickle.load(f)
                f.close()
                
                data_pupil_c_error_un.append(data['pupil_c_error_un'])
                data_pupil_c_error.append(data['pupil_c_error'])
                data_iris_c_error_un.append(data['iris_c_error_un'])
                data_iris_c_error.append(data['iris_c_error'])
                
                if model_name=='ritnet_v2':
                    first_name='Ours w/L'
                if model_name=='ritnet_v6':
                    first_name='RITnet w/L'
                if model_name=='deepvog':
                    first_name='DeepVOG'                 
                if model_name=='ritnet_v4':
                    first_name='Ours ElSeg'
                if model_name=='ritnet_v5':
                    first_name='RITnet ElSeg'
    #            save_filename.append(dataset_name+'/'+model_name+'/'+test_case)
#                save_filename.append(model_name[-2:]+'/'+test_case)
                save_filename.append(first_name)

    
                data_background_iou.append(np.array(data['iou_sample'])[:,0])
                data_iris_iou.append(np.array(data['iou_sample'])[:,1])
                data_pupil_iou.append(np.array(data['iou_sample'])[:,2])
                overall_iou=np.nanmean(np.array(data['iou_sample']),axis=1)
                data_overall_iou.append(overall_iou)
#                data_overall_iou.append(np.array(data['iou']))
                data_iris_el_iou.append(np.array(data_el['data_iris_el_iou']))
                data_pupil_el_iou.append(np.array(data_el['data_pupil_el_iou']))
                data_iris_el_angular_error.append(np.array(data_el['data_iris_el_angular_error_after_ratio_test']))
                data_pupil_el_angular_error.append(np.array(data_el['data_pupil_el_angular_error_after_ratio_test']))


        if include_ellipse_fit_data:
            for model_name in ['ritnet_v4','ritnet_v5']:
                folder_name='giw_e2e'    
                filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
                          model_name+'/'+'opDict_iris_center_estimate_eye_parts.npy'
                data_iris_c_error_un.append(np.load(filename)) 
                filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
                          model_name+'/'+'opDict_pupil_center_estimate_eye_parts.npy'
                data_pupil_c_error_un.append(np.load(filename))
                if model_name=='ritnet_v4':
                    first_name='Ours w/Ef'
                if model_name=='ritnet_v5':
                    first_name='RITnet w/Ef'                
                save_filename.append(first_name)
                  
        if include_deepvog:
            folder_name='giw_e2e'
            model_name='deepvog'
            filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
                      model_name+'/'+'opDict_iris_center_estimate_eye_parts.npy'
            
            data_iris_c_error_un.append([]) 
            filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
                      model_name+'/'+'opDict_pupil_center_estimate_eye_parts.npy'
            data_pupil_c_error_un.append(np.load(filename))
                  
            filename='/media/aaa/hdd/ALL_model/giw_e2e/op/'+dataset_name+'/'+\
                        'deepvog/_0_0opDict_scores.pkl'
            f = open(filename, 'rb')
            data=pickle.load(f)
            f.close()
            
#            data_pupil_c_error_un.append([])
#            data_pupil_c_error.append([])
#            data_iris_c_error_un.append([])
#            data_iris_c_error.append([])
    #            save_filename.append(dataset_name+'/'+model_name+'/'+test_case)
            save_filename.append('DeepVOG')
            
            data_background_iou.append(np.array(data['iou_sample'])[:,0])
            data_iris_iou.append(np.array(data['iou_sample'])[:,2]) #all nan
            data_pupil_iou.append(np.array(data['iou_sample'])[:,1])
#            data_overall_iou.append(np.array(data['iou']))
            overall_iou=np.nanmean(np.array(data['iou_sample']),axis=1)
            data_overall_iou.append(overall_iou)        
        if include_Excuse:
            filename='/media/aaa/hdd/ALL_model/giw_e2e/op/Excuse/'+dataset_name+'_Excuse_distance_resized.npy'
            data_pupil_c_error_un.append(np.load(filename))
            save_filename.append('ExCuSe')
            data_iris_c_error_un.append([])

        if figures_to_plot=='part_3_1':    
            mean_values=np.array([np.nanmean(x) for x in list(data_pupil_c_error_un)])
            save_plot_hist(data_pupil_c_error_un,save_filename,0,4, mean_values,"unnormalized pupil error","uP"+dataset_name+model_name+test_case,dataset_name,gs,counter,0,figures_to_plot)
            #
            mean_values=np.array([np.nanmean(x) for x in list(data_iris_c_error_un)])
            save_plot_hist(data_iris_c_error_un,save_filename,0,4, mean_values,"unnormalized iris error","iP"+dataset_name+model_name+test_case,dataset_name,gs,counter,1,figures_to_plot)
            counter=counter+1
    
        if figures_to_plot=='part_3_2':    
#            mean_values=np.nanmean(np.array(data_pupil_el_iou),axis=1)
            mean_values=np.array([np.nanmean(x) for x in list(data_pupil_el_iou)])
            save_plot(data_pupil_el_iou,save_filename,1,0.6, mean_values,"Pupil ellipse","uP"+dataset_name+model_name+test_case,dataset_name,gs,counter,0)    
            
            mean_values=np.array([np.nanmean(x) for x in list(data_iris_el_iou)])
#            mean_values=np.nanmean(np.array(data_iris_el_iou),axis=1)
            save_plot(data_iris_el_iou,save_filename,1,0.6, mean_values,"Iris ellipse","uP"+dataset_name+model_name+test_case,dataset_name,gs,counter,1)
        
#            mean_values=np.nanmean(np.array(data_pupil_el_angular_error),axis=1)
            mean_values=np.array([np.nanmean(x) for x in list(data_pupil_el_angular_error)])
            save_plot(data_pupil_el_angular_error,save_filename,0,90, mean_values,"Pupil ellipse angular error","uP"+dataset_name+model_name+test_case,dataset_name,gs,counter,2)    
        
#            mean_values=np.nanmean(np.array(data_iris_el_angular_error),axis=1)
            mean_values=np.array([np.nanmean(x) for x in list(data_iris_el_angular_error)])
            save_plot(data_iris_el_angular_error,save_filename,0,90, mean_values,"Iris ellipse angular error","uP"+dataset_name+model_name+test_case,dataset_name,gs,counter,3)    
            counter=counter+1
    
        if figures_to_plot=='part_3_3':   
            if dataset_name=='riteyes_general':
                dataset_name='RITeyes_general'
    
            mean_values=np.nanmean(np.array(data_overall_iou),axis=1)
            save_plot(data_overall_iou,save_filename,1,0.7, mean_values,"Overall iou","mIou"+dataset_name+model_name+test_case,dataset_name,gs,counter,0)
        
            mean_values=np.nanmean(np.array(data_iris_iou),axis=1)
            save_plot(data_iris_iou,save_filename,1,0.7, mean_values,"Iris Per-class accuracy","mIoUi"+dataset_name+model_name+test_case,dataset_name,gs,counter,2)
        
            mean_values=np.nanmean(np.array(data_pupil_iou),axis=1)
            save_plot(data_pupil_iou,save_filename,1,0.7, mean_values,"Pupil Per-class accuracy","mIouP"+dataset_name+model_name+test_case,dataset_name,gs,counter,1)
        
    #        mean_values=np.nanmean(np.array(data_background_iou),axis=1)
    #        save_plot(data_background_iou,save_filename,1,0.9, mean_values,"Per-class accuracy Background","mIoUb"+dataset_name+model_name+test_case,dataset_name,gs,counter,3)
            counter=counter+1
    #gs.tight_layout(fig)
    #fig.savefig('result/iou_scores.png',bbox_inches='tight')
    
    gs.tight_layout(fig)
    fig.savefig('result/'+figures_to_plot+'.png',bbox_inches='tight',dpi=200)

if figures_to_plot=='part_2_1':
    for dataset_name in dataset_names:
        data_pupil_c_error_un=[]
        data_pupil_c_error=[]
        data_iris_c_error_un=[]
        data_iris_c_error=[]
        data_pupil_iou=[]
        data_iris_iou=[]
        data_overall_iou=[]
        data_background_iou=[]
        data_iris_el_iou=[]
        data_pupil_el_iou=[]
        data_iris_el_angular_error=[]
        data_pupil_el_angular_error=[]
        save_filename=[]
        
        folder_names=['giw_e2e','GIW_e2e_temp']
        for folder_name in folder_names:
            if folder_name=='giw_e2e':
                test_cases =['_0_0']
            else:
                test_cases=['']
            for model_name in model_names:
                for test_case in test_cases:
#                  if not (folder_name=='giw_e2e') & (model_name=='ritnet_v5'):
                  filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
                            model_name+'/'+'opDict_iris_center_estimate_eye_parts.npy'
                  
                  data
                  data_iris_c_error.append(np.load(filename)) 
                  filename='/media/aaa/hdd/ALL_model/'+folder_name+'/op/'+dataset_name+'/'+\
                            model_name+'/'+'opDict_pupil_center_estimate_eye_parts.npy'
                  data_pupil_c_error.append(np.load(filename))
#                  save_filename.append(model_name+folder_name)
                  second_name = "ElSeg" if folder_name=='giw_e2e' else "EpSeg"
                  if model_name=='ritnet_v4':
                      first_name='Ours'
                  if model_name=='ritnet_v5':
                      first_name='RITnet'
                  if model_name=='deepvog':
                      first_name='DeepVOG'                 
                  save_filename.append(first_name+'_'+second_name)
         
        if dataset_name=='OpenEDS':
            filename='/media/aaa/hdd/ALL_model/DeepVOG_pretrained_pupil_center_error.npy'
            data_pupil_c_error.append(np.load(filename))
            save_filename.append('DeepVOG_Pretrained')
            data_iris_c_error.append([])
        

        #mean_values=np.nanmean(np.array(data_pupil_c_error),axis=1)
        mean_values=0
        save_plot_hist(data_pupil_c_error,save_filename,1,3, mean_values,"unnormalized pupil error","uP"+dataset_name+model_name+test_case,dataset_name,gs,counter,0,figures_to_plot)
        #
        #mean_values=np.nanmean(np.array(data_iris_c_error),axis=1)
        save_plot_hist(data_iris_c_error,save_filename,1,3, mean_values,"unnormalized iris error","iP"+dataset_name+model_name+test_case,dataset_name,gs,counter,1,figures_to_plot)
        counter=counter+1
    gs.tight_layout(fig)
    fig.savefig('result/'+figures_to_plot+'.png',bbox_inches='tight',dpi=200)

