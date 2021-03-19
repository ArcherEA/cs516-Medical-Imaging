# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:10:22 2020

@author: Yukuan
"""

#assignment1 part1：
import nibabel
import numpy as np
import matplotlib.pyplot as plt
img_part1 = nibabel.loadsave.load('I:/t1.nii')
img_data =img_part1.get_fdata()

def scroll_press(event):
    print(event.button)
    fig = event.canvas.figure
    if fig.view_all == True:
        if fig.view_axial == True:
            ax = fig.axes[0]
            fig.dim=2
        elif fig.view_sagittal == True:
            ax = fig.axes[1]
            fig.dim=0
        elif fig.view_coronal == True:
            ax = fig.axes[2] 
            fig.dim=1
    else:ax = fig.axes[0]
    if ( event.button == 'up' ):
        if fig.dim==0:
            ax.index=(ax.index+1) % fig.imgs.shape[0]
            ax.images[0].set_array(np.rot90(fig.imgs[ax.index,:, :]))
        elif fig.dim==1:
            ax.index=(ax.index+1) % fig.imgs.shape[1]
            ax.images[0].set_array(np.rot90(fig.imgs[:,ax.index,:]))
        elif fig.dim==2:
            ax.index=(ax.index+1) % fig.imgs.shape[2]
            ax.images[0].set_array(np.rot90(fig.imgs[:, :,ax.index]))
    elif( event.button == 'down'):
        if fig.dim==0:
            ax.index=(ax.index-1) % fig.imgs.shape[0]
            ax.images[0].set_array(np.rot90(fig.imgs[ax.index,:, :]))
        elif fig.dim==1:
            ax.index=(ax.index-1) % fig.imgs.shape[1]
            ax.images[0].set_array(np.rot90(fig.imgs[:,ax.index, :]))
        elif fig.dim==2:
            ax.index=(ax.index-1) % fig.imgs.shape[2]
            ax.images[0].set_array(np.rot90(fig.imgs[:, :,ax.index]))
    ax.set_title('slice='+str(ax.index))
    fig.canvas.draw_idle()

def on_key_press(event):
    print(event.key)
    fig = event.canvas.figure
    if fig.view_all == True:
        if fig.view_axial == True:
            ax = fig.axes[0]
            fig.dim=2
        elif fig.view_sagittal == True:
            ax = fig.axes[1]
            fig.dim=0
        elif fig.view_coronal == True:
            ax = fig.axes[2] 
            fig.dim=1
    else:ax = fig.axes[0]
    if (event.key == 'up'  ):
        if fig.dim==0:
            ax.index=(ax.index+1) % fig.imgs.shape[0]
            ax.images[0].set_array(np.rot90(fig.imgs[ax.index,:, :]))
        elif fig.dim==1:
            ax.index=(ax.index+1) % fig.imgs.shape[1]
            ax.images[0].set_array(np.rot90(fig.imgs[:,ax.index,:]))
        elif fig.dim==2:
            ax.index=(ax.index+1) % fig.imgs.shape[2]
            ax.images[0].set_array(np.rot90(fig.imgs[:, :,ax.index]))
    elif(event.key == 'down'):
        if fig.dim==0:
            ax.index=(ax.index-1) % fig.imgs.shape[0]
            ax.images[0].set_array(np.rot90(fig.imgs[ax.index,:, :]))
        elif fig.dim==1:
            ax.index=(ax.index-1) % fig.imgs.shape[1]
            ax.images[0].set_array(np.rot90(fig.imgs[:,ax.index, :]))
        elif fig.dim==2:
            ax.index=(ax.index-1) % fig.imgs.shape[2]
            ax.images[0].set_array(np.rot90(fig.imgs[:, :,ax.index]))
    ax.set_title('slice='+str(ax.index))
    fig.canvas.draw_idle()
    
def axes_enter_event(event):
   print(event.inaxes)
   fig = event.canvas.figure
   ax = fig.axes[0]
   ax1 = fig.axes[1]
   ax2 = fig.axes[2]
   if event.inaxes==ax:
       print('axial view')
       fig.view_axial = True
   elif event.inaxes==ax1:
       print('sagittal view')
       fig.view_sagittal = True
   elif event.inaxes==ax2:
       print('coronal view')
       fig.view_coronal = True

def axes_leave_event(event):
    print(event.inaxes)
    fig = event.canvas.figure
    ax = fig.axes[0]
    ax1 = fig.axes[1]
    ax2 = fig.axes[2]
    if event.inaxes==ax:
        print('axial view')
        fig.view_axial = False
    elif event.inaxes==ax1:
        print('sagittal view')
        fig.view_sagittal = False
    elif event.inaxes==ax2:
        print('coronal view')
        fig.view_coronal = False
        
#i use 3d image to do histogram equalization 
#and it cost a lot of time to get the result 
def hist_eq(img):
    #count pixels number
    img=img.astype(np.int64)
    img_max=np.max(img)
    c=np.zeros(img_max+1,dtype=np.float)
    #store possibility
    p=np.zeros(img_max+1,dtype=np.float)
    #store possibility for new image(after histgram equalization)
    o=np.zeros(img_max+1,dtype=np.float)
    s0=img.shape[0]
    s1=img.shape[1]
    s2=img.shape[2]
    for x in img:
        #count number of the same value pixel
        c[x]+=1
    for i in range(0,c.size): 
        #calculate the possibility of the same value pixel
        p[i]=c[i]/float(img.size) 
    o[0] = p [0]
    for i in range(0,c.size):
        o[i]=o[i-1]+p[i]
    out = np.zeros((s0,s1,s2))
    for x in range(0,s0):
        for y in range(0,s1):
            for z in range(0,s2):
               out[x,y,z]=255*o[img[x,y,z]]
    return out
def viewer(brain,slice=250,view='axial',histeq=False):
    fig =plt.figure()
    fig.imgs=brain
    fig.view_all = False
    if histeq==True:
       fig.imgs=hist_eq(brain) 
    if view =='all':
        #create FLAGS
        fig.view_all = True
        fig.view_axial = False
        fig.view_sagittal = False
        fig.view_coronal = False
        ax = fig.add_subplot(2,2,1)
        ax1 = fig.add_subplot(2,2,2)
        ax2 = fig.add_subplot(2,2,3)
        ax.index=brain.shape[2] // 2
        ax1.index=brain.shape[0] // 2
        ax2.index=brain.shape[1] // 2
        ax.imshow(np.rot90(fig.imgs[:,:,ax.index]))
        ax.set_title('slice='+str(ax.index))
        ax1.imshow(np.rot90(fig.imgs[ax1.index,:,:]))
        ax1.set_title('slice='+str(ax1.index))
        ax2.imshow(np.rot90(fig.imgs[:,ax2.index,:]))
        ax2.set_title('slice='+str(ax2.index))
        fig.canvas.mpl_connect('axes_enter_event' , axes_enter_event)
        fig.canvas.mpl_connect('axes_leave_event' , axes_leave_event)  
    else:
        ax = fig.add_subplot()
        ax.index=slice
    if view=='axial':
        fig.dim=2
        ax.imshow(np.rot90(fig.imgs[:,:,slice%brain.shape[2]]))
        plt.title('slice='+str(slice))
    elif view =='sagittal':
        fig.dim=0
        ax.imshow(np.rot90(fig.imgs[slice%brain.shape[0],:,:]))
        plt.title('slice='+str(slice))
    elif view =='coronal':
        fig.dim =1
        ax.imshow(np.rot90(fig.imgs[:,slice%brain.shape[1],:])) 
        plt.title('slice='+str(slice))  
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('scroll_event', scroll_press)

viewer(img_data,view='sagittal')
#viewer(img_data,view='all',histeq=True)
#viewer(img_data,view='all',histeq=False)
#viewer(img_data,view='axial',slice=250)
#viewer(img_data,view='coronal',slice=250)
