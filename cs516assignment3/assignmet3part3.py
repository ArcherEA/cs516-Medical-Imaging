# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:51:03 2020

@author: Yukuan Hao
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import rotate

'''
preprocessing images
'''
for i in range(1,17):
    fmri = nib.load('I:/image/new_dataset/sub'+str(i)+'/clean_bold.nii.gz');
    #fmri = nib.load('I:/a3_data/bold.nii.gz');
    events = pd.read_csv('I:/image/new_dataset/sub'+str(i)+'/events.tsv',delimiter='\t');
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for j in np.arange(0,events.shape[0]):
        if events[j,3]=='FAMOUS' or events[j,3]=='UNFAMILIAR' or events[j,3]=='SCRAMBLED':
            ts[int(events[j,0])] = 1    
    hrf=pd.read_csv('I:/a3_data/hrf.csv',header=None).to_numpy()
    conved = signal.convolve(ts,hrf.reshape(-1),mode='full')
    conved = conved[0:ts.shape[0]]
    conved = conved[0::2]
    img = fmri.get_data()
    meansub_img = img - np.expand_dims(np.mean(img,3),3)
    meansub_conved = conved - np.mean(conved)
    corrs = np.sum(meansub_img*meansub_conved,3)/(np.sqrt(np.sum(meansub_img*meansub_img,3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))
    #where_are_NaNs = np.isnan(corrs)
    #corrs[where_are_NaNs] = 0
    corrs_nifti=nib.Nifti1Image(corrs, fmri.affine)
    nib.save(corrs_nifti,'I:/image/new_dataset/sub'+str(i)+'/corrs.nii.gz')

'''
extract MNI152 IMAGE with skull
'''
fmri = nib.load('I:/image/part1/a3_data/MNI152_2009_template.nii.gz');
img = fmri.get_data()
img1=img[:,:,:,0,1]
corrs_nifti=nib.Nifti1Image(img1, fmri.affine)
nib.save(corrs_nifti,'I:/image/MNI152.nii.gz')
plt.imshow(img1[:,:,90])
nib.save(corrs_nifti,'I:/a3_data/corrs.nii.gz')


'''
calculate average of 16 corrs
'''
imgs=np.zeros((193,229,193,16))

for i in range(0,16):
    sub1=nib.load('I:/image/corrs/corrs_in_MNI'+str(i+1)+'.nii.gz');
    img=sub1.get_data()
    where_are_NaNs = np.isnan(img)
    img[where_are_NaNs] = 0
    imgs[:,:,:,i]=img
    
img_avg=np.average(imgs,axis=3)


corrs_nifti=nib.Nifti1Image(img_avg, sub1.affine)
nib.save(corrs_nifti,'I:/image/corrs/corrs_avg_in_MNI1.nii.gz')
plt.imshow(img_avg[:,:,90])


