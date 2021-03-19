# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:06:19 2020

@author: Yukuan
"""
import nibabel
import numpy as np
import matplotlib.pyplot as plt
#2a:
def fft(img):
    f=np.fft.fft2(img)
    fshift=np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(np.rot90(img))
    plt.title('raw')
    plt.subplot(122),plt.imshow(np.rot90(magnitude_spectrum))
    plt.title('frequency domain of raw')
    plt.show()
img_part2a = nibabel.loadsave.load('I:/t2.nii')
img2a_data =img_part2a.get_fdata()
sp=img2a_data.shape   
fft(img2a_data[:,:,250])