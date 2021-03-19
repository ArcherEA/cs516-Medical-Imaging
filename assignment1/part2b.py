# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:06:48 2020

@author: Yukuan
"""
import nibabel
import numpy as np
import matplotlib.pyplot as plt
def gaussian_smooth(img):
    sz_x = img.shape[0]
    sz_y = img.shape[1]
    [X, Y] = np.mgrid[0:sz_x, 0:sz_y]
    xpr = X - int(sz_x) // 2
    ypr = Y - int(sz_y) // 2
    fft1=np.fft.fft2(img)
    ft2=np.fft.fftshift(fft1)
    count=1
    for sigma in range(1,25,5):
        gaussfilt = np.exp(-((xpr**2+ypr**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
        plt.figure(1)
        plt.subplot(1,5,count); plt.imshow(np.rot90(gaussfilt));
        plt.title('sigma='+str(sigma))
        plt.figure(2)
        gaussian =ft2*gaussfilt
        plt.subplot(1,5,count);
        plt.imshow(np.rot90(np.float64(np.log(gaussian))));
        plt.title('sigma='+str(sigma))
        ifshift = np.fft.ifftshift(gaussian);
        if1 = np.fft.ifft2(ifshift)
        plt.figure(3)
        plt.subplot(1,5,count);plt.imshow(np.rot90(np.float64(if1)));
        plt.title('sigma='+str(sigma))
        count =count + 1
    plt.show()
img_part2B = nibabel.loadsave.load('I:/swi.nii')
img2B_data =img_part2B.get_fdata()
sp=img2B_data.shape 
gaussian_smooth(img2B_data[:,:,250])