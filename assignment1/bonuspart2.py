# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:02:32 2020

@author: Yukuan
"""
import nibabel
import numpy as np
import matplotlib.pyplot as plt
#bonus
img_bonus = nibabel.loadsave.load('I:/t1.nii')
img_data =img_bonus.get_fdata()
img=img_data[:,:,250]
squarefilter=np.zeros(img.shape)
rows, cols = img.shape
crow = int(rows/2)
ccol=int(cols/2)
squarefilter[crow-100:crow+100,ccol-100:ccol+100]=0.15
squarefilter[crow-80:crow+80,ccol-80:ccol+80]=0.3
squarefilter[crow-60:crow+60,ccol-60:ccol+60]=0.5
squarefilter[crow-40:crow+40,ccol-40:ccol+40]=0.75
squarefilter[crow-20:crow+20,ccol-20:ccol+20]=1
plt.imshow(np.rot90(squarefilter))
fft1=np.fft.fft2(img)
ft2=np.fft.fftshift(fft1)
edge=ft2*squarefilter
eifshift = np.fft.ifftshift(edge);
eif1 = np.fft.ifft2(eifshift)
plt.imshow(np.rot90(np.float64(eif1)));