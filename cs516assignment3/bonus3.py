# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 04:27:13 2020

@author: Yukuan Hao
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import rotate
plt.rcParams.update({'font.size': 20})

fmri = nib.load('I:/image/new_dataset/sub1/clean_bold.nii.gz');

#fmri = nib.load('I:/a3_data/bold.nii.gz');
events = pd.read_csv('I:/image/new_dataset/sub1/events.tsv',delimiter='\t');
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
jhist=np.histogram2d(conved, np.ravel(img[0,0,0,:]),bins=10)
from sklearn import metrics
img_new=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            img_new[i,j,k]=metrics.mutual_info_score(conved, img[i,j,k,:])          
plt.imshow(np.rot90(img_new[:,:,32]))
mi_nifti=nib.Nifti1Image(img_new, fmri.affine)
nib.save(mi_nifti,'I:/image/new_dataset/sub1/mi.nii.gz')
