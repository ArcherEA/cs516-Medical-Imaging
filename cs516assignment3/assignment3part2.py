# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 21:10:47 2020

@author: Yukuan Hao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:32:06 2020

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
fmri = nib.load('I:/image/part1/a3_data/bold.nii.gz');
events = pd.read_csv('I:/image/part1/a3_data/events.tsv',delimiter='\t');
events = events.to_numpy()
tr = fmri.header.get_zooms()[3]
ts = np.zeros(int(tr*fmri.shape[3]))
for j in np.arange(0,events.shape[0]):
    if events[j,3]=='FAMOUS' or events[j,3]=='UNFAMILIAR' or events[j,3]=='SCRAMBLED':
        ts[int(events[j,0])] = 1    
plt.figure()
plt.plot(ts); plt.xlabel('time(seconds)')
plt.figure()
hrf=pd.read_csv('I:/image/part1/a3_data/hrf.csv',header=None).to_numpy()
conved = signal.convolve(ts,hrf.reshape(-1),mode='full')
conved = conved[0:ts.shape[0]]
plt.plot(ts)
plt.plot(conved*3.2,lineWidth=3); plt.xlabel('time(seconds)')
conved = conved[0::2]
img = fmri.get_data()
meansub_img = img - np.expand_dims(np.mean(img,3),3)
meansub_conved = conved - np.mean(conved)
plt.figure()
corrs = np.sum(meansub_img*meansub_conved,3)/(np.sqrt(np.sum(meansub_img*meansub_img,3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))
where_are_NaNs = np.isnan(corrs)
corrs[where_are_NaNs] = 0
plt.imshow(np.rot90(np.max(corrs,axis=2)),vmin=-0.25, vmax=0.25)
plt.colorbar()
plt.figure()
corrs_nifti=nib.Nifti1Image(corrs, fmri.affine)
nib.save(corrs_nifti,'I:/image/part1/a3_data/corrs.nii.gz')
corrs_all=np.zeros((77*7,65*10))
for i in range(0,corrs.shape[2]):
    h=i%10
    v=i//10
    corrs_all[v*77:v*77+77,h*65:h*65+65]=np.rot90(corrs[:,:,i])
plt.imshow(corrs_all,vmin=-0.25, vmax=0.25)
plt.colorbar()
