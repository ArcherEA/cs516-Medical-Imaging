# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 02:04:36 2020

@author: Yukuan Hao
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import rotate
fmri = nib.load('I:/image/new_dataset/sub1/clean_bold.nii.gz');#'+str(j)+'
events = pd.read_csv('I:/image/new_dataset/sub1/events.tsv',delimiter='\t');
events = events.to_numpy()
img = fmri.get_data()
flag1=0
flag2=0
for i in np.arange(0,events.shape[0]):
    if events[i,3]=='FAMOUS':
        flag1+=1
     elif events[i,3]=='UNFAMILIAR':
        flag2+=1
f1img=np.zeros((65,77,67,flag1+1))
f2img=np.zeros((65,77,67,flag2+1))
flag1=0
flag2=0
for i in np.arange(0,events.shape[0]):
    if events[i,3]=='FAMOUS':
        flag1+=1
        f1img[:,:,:,flag1]=img[:,:,:,int((events[i,0]+4.5)/2)]
    elif events[i,3]=='UNFAMILIAR':
        flag2+=1
        f2img[:,:,:,flag2]=img[:,:,:,int((events[i,0]+4.5)/2)]     
t=stats.ttest_ind(f1img,f2img,axis=3)
where_are_NaNs = np.isnan(t[0])
t[0][where_are_NaNs] = 0
where_are_NaNs = np.isnan(t[1])
t[1][where_are_NaNs] = 0
t_nifti=nib.Nifti1Image(t[0], fmri.affine)
nib.save(t_nifti,'I:/image/new_dataset/sub1/t.nii.gz')
plt.imshow(np.rot90(t[0][:,:,40]))
plt.colorbar()

