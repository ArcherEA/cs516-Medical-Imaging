# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:03:07 2020

@author: Yukuan Hao
"""

from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

img1 =imread('H:\\assignment_2_data\I2.jpg')
img2 =imread('H:\\assignment_2_data\J2.jpg')
img1 =imread('H:\\assignment_2_data\I2.png')
img2 =imread('H:\\assignment_2_data\J2.png')
bin_size=50
#part1
def joint_histogram(img1,img2,bin_size=50):
    
    joint_hist_bin=np.zeros((bin_size,bin_size))
    max_1=np.max(img1).astype(int)
    max_2=np.max(img2).astype(int)
    min_1=np.min(img1).astype(int)
    min_2=np.min(img1).astype(int)
    joint_hist=np.zeros((max_1+1,max_2+1))###suppose the minimum equal to zero
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            joint_hist[np.int(img1[i,j]),np.int(img2[i,j])]+=1
    plt.figure()
    plt.xlim((min_1, max_1))
    plt.ylim((min_2, max_2))
    plt.imshow(np.log(joint_hist))
    x_bin=(joint_hist.shape[0]+1)/bin_size
    y_bin=(joint_hist.shape[1]+1)/bin_size
    for i in range(joint_hist.shape[0]):
        for j in range(joint_hist.shape[1]):
            joint_hist_bin[np.int(i//x_bin),np.int(j//y_bin)]+=joint_hist[i,j] 
    plt.figure()
    plt.ylim((min_1//x_bin, max_1//x_bin))
    plt.ylim((min_2//y_bin, max_2//y_bin))
    plt.imshow(np.log(joint_hist_bin))
    return joint_hist
joint_histogram(img1,img2,bin_size=50)

#2a
def ssd(img1,img2):
    img1_1=img1.astype(int)
    img2_2=img2.astype(int)
    minus=img1_1-img2_2
    square=minus*minus
    ssd_result=np.sum(square)
    return ssd_result
ssd(img1,img2)
#2b
def corr(img1,img2):
    img1_1=img1.astype(int)
    img2_2=img2.astype(int)
    numerator=np.sum((img1_1-np.average(img1_1))*(img2_2-np.average(img2_2)))
    denominator=np.sqrt(np.sum((img1_1-np.average(img1_1))**2))*np.sqrt(np.sum((img2_2-np.average(img2_2))**2))
    p=numerator/denominator #####result
    return p
corr(img1,img2)
####compare with library function
'''
img1_1=img1.astype(int)
img2_2=img2.astype(int)
import scipy.stats
c=scipy.stats.pearsonr(img1_1.reshape(-1),img2_2.reshape(-1))
cc=np.corrcoef(img1_1,img2_2)
print((img1_1-np.average(img1_1)))
print((img2_2-np.average(img2_2)))
'''
#2c
def mutual_information(img1,img2):
    joint_hist=joint_histogram(img1,img2)
    total_pixel=np.sum(joint_hist)
    normal_jhist=joint_hist/total_pixel
    mi=np.zeros(1).astype(float)
    for i in range (normal_jhist.shape[0]):
        px=np.sum(normal_jhist[i,:])
        for j in range (normal_jhist.shape[1]):
            py=np.sum(normal_jhist[:,j])
            if normal_jhist[i,j]>0:
                mi+=normal_jhist[i,j]*np.log(normal_jhist[i,j]/(px*py))
    print(mi)
    return mi
mutual_information(img1,img2)
#part3


###create 3d point cloud
fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')
x,y,z = 20,20,4
[X, Y, Z] = np.mgrid[0:x,0:y,0:z]
X1=X.reshape(-1)
Y1=Y.reshape(-1)
Z1=Z.reshape(-1)
sumxyz=np.ones((4,1600))
sumxyz[0,:]=X1
sumxyz[1,:]=Y1
sumxyz[2,:]=Z1
ax.scatter(sumxyz[0,:],sumxyz[1,:],sumxyz[2,:],c='black',marker='o')
plt.show()
####rigid transform
p,q,r,s=1,2,3,5
th,om,ph=15,15,15
def rigid_transform(th,om,ph,p,q,r,mat):
    theta = np.pi/180*th
    omega=np.pi/180*om
    phi=np.pi/180*ph
    ryz=np.zeros((4,4))
    rxz=np.zeros((4,4))
    rxy=np.zeros((4,4))
    translate=np.zeros((4,4))  
    ryz[0,0]=1
    ryz[1,1]=np.cos(theta)
    ryz[1,2]=-np.sin(theta)
    ryz[2,1]=np.sin(theta)
    ryz[2,2]=np.cos(theta)
    ryz[3,3]=1
    rxz[0,0]=np.cos(omega)
    rxz[0,2]=np.sin(omega)
    rxz[1,1]=1
    rxz[2,0]=-np.sin(omega)
    rxz[2,2]=np.cos(omega)
    rxz[3,3]=1
    rxy[0,0]=np.cos(phi)
    rxy[0,1]=-np.sin(phi)
    rxy[1,0]=np.sin(phi)
    rxy[1,1]=np.cos(phi)
    rxy[2,2]=1
    rxy[3,3]=1
    translate[0,0]=1
    translate[1,1]=1
    translate[2,2]=1
    translate[3,3]=1
    translate[0,3]=p
    translate[1,3]=q
    translate[2,3]=r
    result=np.ones((4,1600))
    for i in range (mat.shape[1]):
        result[:,i]=np.dot(mat[:,i],ryz)
        result[:,i]=np.dot(result[:,i],rxz)
        result[:,i]=np.dot(result[:,i],rxy)
        result[:,i]=np.dot(result[:,i],translate)
      
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(result[0,:],result[1,:],result[2,:],c='r',marker='o')
    plt.show()
rigid_transform(th,om,ph,p,q,r,sumxyz)

def affine_transform(s,th,om,ph,p,q,r,mat):
    theta = np.pi/180*th
    omega=np.pi/180*om
    phi=np.pi/180*ph
    ryz=np.zeros((4,4))
    rxz=np.zeros((4,4))
    rxy=np.zeros((4,4))
    translate=np.zeros((4,4))
    scaling=np.zeros((4,4))
    ryz[0,0]=1
    ryz[1,1]=np.cos(theta)
    ryz[1,2]=-np.sin(theta)
    ryz[2,1]=np.sin(theta)
    ryz[2,2]=np.cos(theta)
    ryz[3,3]=1
    rxz[0,0]=np.cos(omega)
    rxz[0,2]=np.sin(omega)
    rxz[1,1]=1
    rxz[2,0]=-np.sin(omega)
    rxz[2,2]=np.cos(omega)
    rxz[3,3]=1
    rxy[0,0]=np.cos(phi)
    rxy[0,1]=-np.sin(phi)
    rxy[1,0]=np.sin(phi)
    rxy[1,1]=np.cos(phi)
    rxy[2,2]=1
    rxy[3,3]=1
    translate[0,0]=1
    translate[1,1]=1
    translate[2,2]=1
    translate[3,3]=1
    translate[0,3]=p
    translate[1,3]=q
    translate[2,3]=r
    scaling[0,0]=s
    scaling[1,1]=s
    scaling[2,2]=s
    
    result=np.ones((4,1600))
    for i in range (mat.shape[1]):
        result[:,i]=np.dot(mat[:,i],ryz)
        result[:,i]=np.dot(result[:,i],rxz)
        result[:,i]=np.dot(result[:,i],rxy)
        result[:,i]=np.dot(result[:,i],translate)
        result[:,i]=np.dot(result[:,i],scaling)
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(result[0,:],result[1,:],result[2,:],c='r',marker='o')
    plt.show()
affine_transform(s,th,om,ph,p,q,r,sumxyz)

def rigid_transform3(mat):
    
    ryz=np.zeros((4,4))
    
    #scaling=np.zeros((4,4))
    ryz[0,0]=0.7182
    ryz[0,1]=-1.3727
    ryz[0,2]=0.5660
    ryz[0,3]=1.8115
    ryz[1,0]=-1.9236
    ryz[1,1]=-4.6556
    ryz[1,2]=-2.5512
    ryz[1,3]=0.2873
    ryz[2,0]=-0.6426
    ryz[2,1]=-1.7985
    ryz[2,2]=-1.6285
    ryz[2,3]=0.7404
    ryz[3,0]=0
    ryz[3,1]=0
    ryz[3,2]=0
    ryz[3,3]=1
    
    #scaling[0,0]=s
    #scaling[1,1]=s
    #scaling[2,2]=s
    
    result=np.ones((4,1600))
    for i in range (mat.shape[1]):
        result[:,i]=np.dot(mat[:,i],ryz)
        
       # result[:,i]=np.dot(result[:,i],scaling)
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(result[0,:],result[1,:],result[2,:],c='r',marker='o')
    ax.scatter(mat[0,:],mat[1,:],mat[2,:],c='black',marker='o')
    plt.show()
rigid_transform3(mat)   
###figure 1   
#ryz[0,0]=0.9045
#ryz[0,1]=-0.3847
#ryz[0,2]=-0.1840
#ryz[0,3]=10
#ryz[1,0]=0.2939
#ryz[1,1]=0.8750
#ryz[1,2]=-0.3847
#ryz[1,3]=10
#ryz[2,0]=0.3090
#ryz[2,1]=0.2939
#ryz[2,2]=0.9045
#ryz[2,3]=10
#ryz[3,0]=0
#ryz[3,1]=0
#ryz[3,2]=0
#ryz[3,3]=1    
###figure2
#ryz[0,0]=0
#ryz[0,1]=-0.2598
#ryz[0,2]=0.15
#ryz[0,3]=-3
#ryz[1,0]=0
#ryz[1,1]=-0.15
#ryz[1,2]=-0.2598
#ryz[1,3]=1.5
#ryz[2,0]=0.3000
#ryz[2,1]=-0
#ryz[2,2]=0
#ryz[2,3]=0
#ryz[3,0]=0
#ryz[3,1]=0
#ryz[3,2]=0
#ryz[3,3]=1
    
###figure3
#ryz[0,0]=0.7182
#ryz[0,1]=-1.3727
#ryz[0,2]=0.5660
#ryz[0,3]=1.8115
#ryz[1,0]=-1.9236
#ryz[1,1]=-4.6556
#ryz[1,2]=-2.5512
#ryz[1,3]=0.2873
#ryz[2,0]=-0.6426
#ryz[2,1]=-1.7985
#ryz[2,2]=-1.6285
#ryz[2,3]=0.7404
#ryz[3,0]=0
#ryz[3,1]=0
#ryz[3,2]=0
#ryz[3,3]=1
##useful variables 
##useful variables 
p,q=20.0,-40.0
img1=imread('H:\\assignment_2_data\BrainMRI_1.jpg')

###part4 a
def translation(img,p,q):
    img=img.astype(float)
    size_x=img.shape[0]
    size_y=img.shape[1]
    x = np.linspace(0,size_x,size_y)
    y = np.linspace(0,size_x,size_y)
    f = interp2d(x+p,y+q,img,kind='cubic',fill_value=0)
    img1=f(x,y)
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    ax1.set_title("before translate")
    ax1.imshow(img)
    ax2.set_title("after translate ")
    ax2.imshow(img1)
    return img1
tran=translation(img1,p,q)

def ssd(img1,img2):
    minus=img1.astype(float)-img2.astype(float)
    square=minus*minus
    ssd_result=np.sum(square)
    return ssd_result
####part 4 b
img1 =imread('H:\\assignment_2_data\BrainMRI_1.jpg').astype(float)
img2 =imread('H:\\assignment_2_data\BrainMRI_4.jpg').astype(float)

def Lucas_Kanade(img2,img1,iteration=200):
    u=np.zeros((2,1))
    size_x=img2.shape[0]
    size_y=img2.shape[1]
    x = np.linspace(0,size_x,size_y)
    y = np.linspace(0,size_x,size_y)
    M=np.zeros((2,2))
    b=np.zeros((2,1))
    result=np.zeros(iteration)
    for i in range(0,iteration):
        f = interp2d(x+u[0],y+u[1],img2,kind='cubic',fill_value=0)
        trans=f(x,y)
        result[i]=ssd(trans,img1)
        #plt.imshow(der_x)
        f_x = interp2d(x-1,y,trans,kind='cubic',fill_value=0)
        f_x1 = interp2d(x+1,y,trans,kind='cubic',fill_value=0)
        f_y = interp2d(x,y-1,trans,kind='cubic',fill_value=0)
        f_y1 = interp2d(x,y+1,trans,kind='cubic',fill_value=0)
        der_x=(f_x(x,y)-f_x1(x,y))/2
        der_y=(f_y(x,y)-f_y1(x,y))/2
        it=trans-img1
        M[0,0]=np.sum(der_x**2)
        M[0,1]=np.sum(der_x*der_y)
        M[1,0]=np.sum(der_y*der_x) #####use np.dot or *?not sure
        M[1,1]=np.sum(der_y**2)
        b[0]=-np.sum(der_x*it)
        b[1]=-np.sum(der_y*it)
        u=u-np.dot(np.linalg.inv(M),b)
    fig=plt.figure()
    ax1=fig.add_subplot(131)
    ax1.set_title("input image1")
    ax1.imshow(img2)
    ax2=fig.add_subplot(132)
    ax2.set_title("input image2")
    ax2.imshow(img1)
    ax3=fig.add_subplot(133)
    ax3.set_title("translate image")
    ax3.imshow(trans)
    plt.figure()
    plt.plot(result)
    print(u)
    return result,u
Lucas_Kanade(img1,img2)

cers =imread('H:\\assignment_2_data\BrainMRI_1.jpg')
img=imread('H:\\assignment_2_data\BrainMRI_1.jpg')
th=15
p=20
q=50
def rotationtranslate(img,th,p,q,show=True):
    u=np.zeros((2,1))
    u[0]=p
    u[1]=q
    sz_x = img.shape[0]
    sz_y = img.shape[1]
    [X,Y] = np.mgrid[0:sz_x,0:sz_y]
    X1=X.reshape(-1)
    Y1=Y.reshape(-1)
    rotmat = np.zeros([3,3])
    new_points=np.ones([3,img.shape[0]*img.shape[1]])
    theta = np.pi/180*th
    new_points[0,:]=X1
    new_points[1,:]=Y1

    rotmat[0,0] = np.cos(theta)
    rotmat[0,1] = -np.sin(theta)
    rotmat[1,0] = np.sin(theta)
    rotmat[1,1] = np.cos(theta)
    rotmat[0,2] = u[0]
    rotmat[1,2] = u[1]
    rotmat[2,2] = 1
    new_point = np.matmul(rotmat,new_points)
    new_point_transpose=np.zeros([img.shape[0]*img.shape[1],2])
    new_point_transpose[:,0]=new_point[0].T
    new_point_transpose[:,1]=new_point[1].T

    new_points[0]=new_point[0]
    new_points[1]=new_point[1]
    value=img.reshape(-1)
    rotimg=griddata(new_point_transpose,value,(X,Y),fill_value=0,method='cubic')
    if show:
        fig=plt.figure()
        ax1=fig.add_subplot(121)
        ax2=fig.add_subplot(122)
        ax1.set_title("before rotate")
        ax1.imshow(img)
        ax2.set_title("after rotate ")
        ax2.imshow(rotimg)
    return rotimg
rotationtranslate(cers,th,p,q)


def rotation(img,th,show=True):
    sz_x = img.shape[0]
    sz_y = img.shape[1]
    [X,Y] = np.mgrid[0:sz_x,0:sz_y]
    X1=X.reshape(-1)
    Y1=Y.reshape(-1)
    rotmat = np.zeros([2,2])
    theta = np.pi/180*th
    rotmat[0,0] = np.cos(theta)
    rotmat[0,1] = -np.sin(theta)
    rotmat[1,0] = np.sin(theta)
    rotmat[1,1] = np.cos(theta)
    new_point = np.matmul(rotmat,(X1,Y1))
    value=img.reshape(-1)
    rotimg=griddata(new_point.T,value,(X,Y),fill_value=0,method='cubic')
    if show:
        fig=plt.figure()
        ax1=fig.add_subplot(121)
        ax2=fig.add_subplot(122)
        ax1.set_title("before rotate")
        ax1.imshow(img)
        ax2.set_title("after rotate ")
        ax2.imshow(rotimg)
    return rotimg
rotation(cers,th)
####part4 d
img1 =imread('H:\\assignment_2_data\BrainMRI_1.jpg').astype(float)
img2 =imread('H:\\assignment_2_data\BrainMRI_3.jpg').astype(float)
img1 =imread('H:\\assignment_2_data\J5.jpg').astype(float)
img2 =imread('H:\\assignment_2_data\J6.jpg').astype(float)
def ssd(img1,img2):
    minus=img1.astype(float)-img2.astype(float)
    square=minus*minus
    ssd_result=np.sum(square)
    return ssd_result
def mini_ssd(img1,img2,iteration=30):
    th=0
    theta = np.pi/180*th
    size_x=img2.shape[0]
    size_y=img2.shape[1]
    [X,Y] = np.mgrid[0:size_x,0:size_y]
    x = np.linspace(0,size_x,size_y)
    y = np.linspace(0,size_x,size_y)
    rotimg=rotation(img1,th,show=False)
    theta = np.pi/180*th
    result=np.zeros(iteration)
    for i in range(0,iteration):
        #plt.imshow(rotimg)
        result[i]=ssd(rotimg,img2)
        #plt.imshow(der_x)
        f_x = interp2d(x-1,y,rotimg,kind='cubic',fill_value=0)
        f_x1 = interp2d(x+1,y,rotimg,kind='cubic',fill_value=0)
        f_y = interp2d(x,y-1,rotimg,kind='cubic',fill_value=0)
        f_y1 = interp2d(x,y+1,rotimg,kind='cubic',fill_value=0)
        der_x=(f_x(x,y)-f_x1(x,y))/2*(X*(-np.sin(theta))-Y*(np.cos(theta)))
        der_y=(f_y(x,y)-f_y1(x,y))/2*(X*(np.cos(theta))-Y*(np.sin(theta)))
        it=rotimg-img2
        th-=2*np.sum(it*(der_x+der_y))*0.000000002###0.000000002 FOR BRAINMRI 0.0000000002 for I5，I6
        rotimg=rotation(img1,th,show=False)
        theta = np.pi/180*th
    fig=plt.figure()
    ax1=fig.add_subplot(221)
    ax1.set_title("img1")
    ax1.imshow(img1)
    ax2=fig.add_subplot(222)
    ax2.set_title("img2")
    ax2.imshow(img2)
    ax3=fig.add_subplot(223)
    ax3.set_title("registration image")
    ax3.imshow((rotimg+img2)/2)
    ax4=fig.add_subplot(224)
    ax4.set_title("image1 in image2")
    ax4.imshow((img1+img2)/2)
    plt.figure()
    plt.plot(result)
    return result,th
mini_ssd(img2,img1,iteration=30)
###4e
def mini_ssd_2(img1,img2,iteration=30):
    th=0
    theta = np.pi/180*th
    size_x=img2.shape[0]
    size_y=img2.shape[1]
    [X,Y] = np.mgrid[0:size_x,0:size_y]
    x = np.linspace(0,size_x,size_y)
    y = np.linspace(0,size_x,size_y)
    p=0
    q=0
    rotimg=rotationtranslate(img1,th,p,q,show=False)
    theta = np.pi/180*th
    result=np.zeros(iteration)
    for i in range(0,iteration):
        
        #plt.imshow(rotimg)
        result[i]=ssd(rotimg,img2)
        #plt.imshow(der_x)
        f_x = interp2d(x-1,y,rotimg,kind='cubic',fill_value=0)
        f_x1 = interp2d(x+1,y,rotimg,kind='cubic',fill_value=0)
        f_y = interp2d(x,y-1,rotimg,kind='cubic',fill_value=0)
        f_y1 = interp2d(x,y+1,rotimg,kind='cubic',fill_value=0)
        der_xangle=(f_x(x,y)-f_x1(x,y))/2*(X*(-np.sin(theta))-Y*(np.cos(theta)))
        der_yangle=(f_y(x,y)-f_y1(x,y))/2*(X*(np.cos(theta))-Y*(np.sin(theta)))
        it=rotimg-img2
        der_x=(f_x(x,y)-f_x1(x,y))/2#*((-np.sin(theta))-(np.cos(theta)))
        der_y=(f_y(x,y)-f_y1(x,y))/2#*((np.cos(theta))-(np.sin(theta)))
        th-=2*np.sum(it*(der_xangle+der_yangle))*0.000000005
        p+=2*np.sum(it*der_x)*0.000001
        q+=2*np.sum(it*der_y)*0.000001
        rotimg=rotationtranslate(img1,th,p,q,show=False)
        theta = np.pi/180*th         
    fig=plt.figure()
    ax1=fig.add_subplot(221)
    ax1.set_title("img1")
    ax1.imshow(img1)
    ax2=fig.add_subplot(222)
    ax2.set_title("img2")
    ax2.imshow(img2)
    ax3=fig.add_subplot(223)
    ax3.set_title("registration image")
    ax3.imshow((rotimg+img2)/2)
    ax4=fig.add_subplot(224)
    ax4.set_title("image1 in image2")
    ax4.imshow((img1+img2)/2)
    plt.figure()
    plt.plot(result)
    
    return result,th,p,q
mini_ssd_2(img2,img1,iteration=50)

