# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:07:19 2020

@author: Yukuan
"""
import nibabel
import numpy as np
import matplotlib.pyplot as plt
t1= nibabel.loadsave.load('I:/t1.nii')
t2= nibabel.loadsave.load('I:/t2.nii')
swi= nibabel.loadsave.load('I:/swi.nii')
bold= nibabel.loadsave.load('I:/bold.nii')
tof= nibabel.loadsave.load('I:/tof.nii')
t1_data=t1.get_fdata()
t2_data=t2.get_fdata()
swi_data=swi.get_fdata()
bold_data=bold.get_fdata()
tof_data=tof.get_fdata()
#2c:
def viewer2c(img,sigma=21):
    sz_x = img.shape[0]
    sz_y = img.shape[1]
    [X, Y] = np.mgrid[0:sz_x, 0:sz_y]
    xpr = X - int(sz_x) // 2
    ypr = Y - int(sz_y) // 2
    fft1=np.fft.fft2(img)
    ft2=np.fft.fftshift(fft1)
    gaussfilt = np.exp(-((xpr**2+ypr**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
    edgefilt = 1-gaussfilt
    edge=ft2*edgefilt
    gaussian =ft2*gaussfilt
    eifshift = np.fft.ifftshift(edge);
    eif1 = np.fft.ifft2(eifshift)
    gifshift = np.fft.ifftshift(gaussian);
    gif1 = np.fft.ifft2(gifshift)
    plt.subplot(2,2,1); plt.imshow(np.rot90(img));
    plt.title('raw')
    plt.subplot(2,2,2);
    plt.imshow(np.rot90(20*np.log(np.abs(ft2))));
    plt.title('fft')
    plt.subplot(2,2,3);plt.imshow(np.rot90(np.float64(eif1)));
    plt.title('edge enhancement')
    plt.subplot(2,2,4);plt.imshow(np.rot90(np.float64(gif1)));
    plt.title('gaussian smoothing')
    plt.show()
    return gif1 #here is useful to do the bonus part,otherwise it is useless
gi=viewer2c(img_data[:,:,250])
plt.figure(1)
viewer2c(t1_data[:,:,250])
plt.figure(2)
viewer2c(t2_data[:,:,250])
plt.figure(3)
viewer2c(swi_data[:,:,250])
plt.figure(4)
viewer2c(bold_data[:,:,18])
plt.figure(5)
viewer2c(tof_data[:,:,81])

def viewer2copt(img,r=20):
    rows, cols = img.shape
    crow = int(rows/2)
    ccol=int(cols/2)
    smoothfilt = np.zeros(img.shape)
    edgefilt = np.ones(img.shape)
    for i in range(0,rows):
        for j in range(0,cols):
            if np.sqrt((i-crow)**2+(j-ccol)**2)<r:
                smoothfilt[i][j]=1
                edgefilt[i][j]=0
    fft1=np.fft.fft2(img)
    ft2=np.fft.fftshift(fft1)
    edge=ft2*edgefilt
    smooth =ft2*smoothfilt
    eifshift = np.fft.ifftshift(edge);
    eif1 = np.fft.ifft2(eifshift)
    sifshift = np.fft.ifftshift(smooth);
    sif1 = np.fft.ifft2(sifshift)
    plt.subplot(2,2,1); plt.imshow(np.rot90(img));
    plt.title('raw')
    plt.subplot(2,2,2);
    plt.imshow(np.rot90(20*np.log(np.abs(ft2))));
    plt.title('fft')
    plt.subplot(2,2,3);plt.imshow(np.rot90(np.float64(eif1)));
    plt.title('edge enhancement')
    plt.subplot(2,2,4);plt.imshow(np.rot90(np.float64(sif1)));
    plt.title('smoothing')
    plt.show()
    return eif1#here is useful to do the bonus part,otherwise it is useless
plt.figure(6)
viewer2copt(t1_data[:,:,250])
plt.figure(7)
viewer2copt(t2_data[:,:,250])
plt.figure(8)
viewer2copt(swi_data[:,:,250])
plt.figure(9)
viewer2copt(bold_data[:,:,18])
plt.figure(10)
viewer2copt(tof_data[:,:,81])