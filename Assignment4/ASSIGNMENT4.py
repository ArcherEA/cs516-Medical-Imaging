import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
from scipy import signal
T1=imread('I:/cs516/a4/t1.png') #(44:64,312:332)(63:83,263:283)
Flair=imread('I:/cs516/a4/flair.png')#noise patch(43:63,440:460) (74:94,384:404)
T1_v2=imread('I:/cs516/a4/t1_v2.png')#(3:23,207:227) (35:55,179:199)
T1_v3=imread('I:/cs516/a4/t1_v3.png')#(50:70,203:223)(80:100,168:188)
T2=imread('I:/cs516/a4/t2.png')#(43:63,314:334) (71:91,:273:293)
plt.imshow(T1_v2)

def bilateral_filter(img,kernel_size=5):
    img=img.astype(int)
    output=np.zeros(img.shape)
    d =int ((kernel_size-1)/2)
    row=img.shape[0]
    col=img.shape[1]
    sigmaspace=(kernel_size/2-1)*0.3+0.8
    pix_avg=np.mean(img)
    pix_max=np.max(img)
    sigmacolorsquare=np.sum((img-pix_avg)*(img-pix_avg))/(row*col)
    for i in range(d,row-d):
        for j in range(d,col-d):
            weightsum=0
            filtervalue=0
            for k in range (-d,d+1):
                for l in range (-d,d+1):
                    distance_square=k*k+l*l
                    value_square=(img[i][j]-img[i+k][j+l])*\
                        (img[i][j]-img[i+k][j+l])
                    weight=np.exp(-1 * (distance_square / \
                                        (2 * sigmaspace*sigmaspace)+ \
                                            value_square / (2 * sigmacolorsquare)))
                    weightsum += weight
                    filtervalue += (weight*img[i+k][j+l])
            output[i][j]=filtervalue/weightsum   
    return output
T1_den2=bilateral_filter(T1,5)
Flair_den2=bilateral_filter(Flair,5)
T1v2_den2=bilateral_filter(T1_v2,5)
T1v3_den2=bilateral_filter(T1_v3,5)
T2_den2=bilateral_filter(T2,5)

def NLmeansfilter(Image, h, templateWindowSize,searchWindowSize):
    f = int(templateWindowSize/2)
    t = int(searchWindowSize/2)
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))
    kernel=kernel/kernel.sum()
    PaddedImg = np.pad(Image,t+f,'symmetric')
    padlength=t+f
    DenoisedImg=np.zeros((Image.shape[0],Image.shape[1]))
    height, width = Image.shape[:2]
    Image_wd = PaddedImg[padlength-f:padlength+f+height, padlength-f:padlength+f+width]
    average = np.zeros(Image.shape)
    sweight = np.zeros(Image.shape)
    wmax =  np.zeros(Image.shape)
    for i in range(-t, t+1):
            for j in range(-t, t+1):
                if i==0 and j==0:
                    continue
                PaddedImg1 = PaddedImg[padlength+i-f:padlength+i+f+height, padlength+j-f:padlength+j+f+width]
    #           w = np.exp(-cv2.filter2D((PaddedImg1 - T1_wd)**2, -1, kreturn)/h**2)[f:f+height, f:f+width]
                b=(PaddedImg1 - Image_wd)**2
                c=kernel
                dist=signal.convolve2d(b,c,boundary='symm',mode='same')
                w = np.exp(-dist/h**2)[f:f+height, f:f+width]
                sweight += w
                wmax = np.maximum(wmax, w)
                average += (w*PaddedImg1[f:f+height, f:f+width])
    DenoisedImg=(average+wmax*Image)/(sweight+wmax)
    return DenoisedImg
#Denoised image
T1_den=NLmeansfilter(T1,4,5,11)
Flair_den=NLmeansfilter(Flair,4,5,11)
T1v2_den=NLmeansfilter(T1_v2,5,5,11)
T1v3_den=NLmeansfilter(T1_v3,4,5,11)
T2_den=NLmeansfilter(T2,4,5,11)

#method noise non local means
mtddenT1=T1-T1_den
mtddenFlair=Flair-Flair_den
mtddenT1v2=T1_v2-T1v2_den
mtddenT1v3=T1_v3-T1v3_den
mtddenT2=T2-T2_den
#bilateral filter
mtddenT12=T1-T1_den2
mtddenFlair2=Flair-Flair_den2
mtddenT1v22=T1_v2-T1v2_den2
mtddenT1v32=T1_v3-T1v3_den2
mtddenT22=T2-T2_den2

'''
#cv2 Nlm
R1 = cv2.fastNlMeansDenoising(T1, None, 20, 5, 11)
R_Flair = cv2.fastNlMeansDenoising(Flair, None, 20, 5, 11)
R1v2 = cv2.fastNlMeansDenoising(T1_v2, None, 20, 5, 11)
R1v3 = cv2.fastNlMeansDenoising(T1_v3, None, 20, 5, 11)
R2 = cv2.fastNlMeansDenoising(T2, None, 20, 5, 11)
'''
def snr(A, B):
    
    return np.mean(B.astype(np.float))/np.std(A.astype(np.float))
#10*np.log(255*255.0/(((A.astype(np.float)-B)**2).mean()))/np.log(10)
'''
#snr of CV2 nlm
snr_R1=snr(T1,R1)
snr_RFlair=snr(Flair,R_Flair)
snr_R1v2=snr(T1_v2,R1v2)
snr_R1v3=snr(T1_v3,R1v3)
snr_R2=snr(T2,R2)
'''


snr_T1=snr(T1[320:330,62:72],T1[290:300,70:80])
snr_TFlair=snr(Flair[440:460,30:50],Flair[405:425,90:110])
snr_T1v2=snr(T1_v2[212:222,22:32],T1_v2[196:206,40:50])
snr_T1v3=snr(T1_v3[210:220,60:70],T1_v3[190:200,100:110])
snr_T2=snr(T2[302:312,50:60],T2[260:270,70:80])


#snr of denoised non local means
snr_T1d=snr(T1_den[320:330,62:72],T1_den[290:300,70:80])
snr_TFlaird=snr(Flair_den[440:460,30:50],Flair_den[405:425,90:110])
snr_T1v2d=snr(T1v2_den[212:222,22:32],T1v2_den[196:206,40:50])
snr_T1v3d=snr(T1v3_den[210:220,60:70],T1v3_den[190:200,100:110])
snr_T2d=snr(T2_den[302:312,50:60],T2_den[260:270,70:80])
#bilateral filter
snr_T1d2=snr(T1_den2[320:330,62:72],T1_den2[290:300,70:80])
snr_TFlaird2=snr(Flair_den2[440:460,30:50],Flair_den2[405:425,90:110])
snr_T1v2d2=snr(T1v2_den2[212:222,22:32],T1v2_den2[196:206,40:50])
snr_T1v3d2=snr(T1v3_den2[210:220,60:70],T1v3_den2[190:200,100:110])
snr_T2d2=snr(T2_den2[302:312,50:60],T2_den2[260:270,70:80])



def imageshow(I,Idenoised,Imetden):
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('origin')
    plt.imshow(I)
    
    plt.subplot(1,3,2)
    plt.title('denoised')
    plt.imshow(Idenoised)
    
    plt.subplot(1,3,3)
    plt.title('method noise')
    plt.imshow(Imetden)
    
imageshow(T1,T1_den,mtddenT1)
imageshow(T1_v2,T1v2_den,mtddenT1v2)
imageshow(T1_v3,T1v3_den,mtddenT1v3)
imageshow(T2,T2_den,mtddenT2)
imageshow(Flair,Flair_den,mtddenFlair)
imageshow(T1,T1_den2,mtddenT12)
imageshow(T1_v2,T1v2_den2,mtddenT1v22)
imageshow(T1_v3,T1v3_den2,mtddenT1v32)
imageshow(T2,T2_den2,mtddenT22)
imageshow(Flair,Flair_den2,mtddenFlair2)

#
def otsu(img):
    x=img.shape[0]
    y=img.shape[1]
    histogram=np.zeros(256)
    for i in range (0,x):
        for j in range (0,y):
            histogram[int(img[i][j])]+=1 
    size=x*y
    u=float(0)
    for i in range (0,256):
        histogram[i]=histogram[i]/size
        u+=i*histogram[i]      
    threshold=0
    maxvariance=float(0)
    w0=float(0)
    avgvalue=0
    for i in range (0,256):
        w0+=histogram[i]
        avgvalue+=i * histogram[i]
        t=float(avgvalue/w0-u)
        variance = float(t * t * w0 /(1 - w0))
        if variance>maxvariance:
            maxvariance=variance
            threshold=i
    return threshold

def otsu_d(img,threshold1,threshold2):#threshold1<threshold2
    if threshold1>threshold2:
        temp=threshold1
        threshold1=threshold2
        threshold2=threshold1
    x=img.shape[0]
    y=img.shape[1]
    histogram=np.zeros(256)
    for i in range (0,x):
        for j in range (0,y):
            histogram[int(img[i][j])]+=1
    u=float(0)
    histogram1=histogram[threshold1:threshold2]
    size=np.sum(histogram1)
    for i in range (0,threshold2-threshold1):
        histogram1[i]=histogram1[i]/size
        u+=i*histogram1[i]      
    threshold=0
    maxvariance=float(0)
    w0=float(0)
    avgvalue=0
    for i in range (0,threshold2-threshold1):
        w0+=histogram1[i]
        avgvalue+=i * histogram1[i]
        t=float(avgvalue/w0-u)
        variance = float(t * t * w0 /(1 - w0))
        if variance>maxvariance:
            maxvariance=variance
            threshold=i
    return threshold+threshold1

tT1=otsu(T1)
tT1_v2=otsu(T1_v2)
tT1_v3=otsu(T1_v3)
tT2=otsu(T2)
tFlair=otsu(Flair)
def create_binary(img,threshold):
    binary=np.zeros(img.shape)
    binary[:]=img[:]
    binary[np.where(binary<threshold)]=0
    binary[np.where(binary>=threshold)]=1
    return binary
def create_binary_2(img,t1,t2):
    binary=np.zeros(img.shape)
    binary[:]=img[:]
    binary[np.where(binary<t1)]=0
    binary[np.where(binary>t2)]=0
    binary[np.where(binary>=t1)]=1
    
    return binary
##### ADDING OTSU SHOW
tT1=otsu(T1)
tOTSUT1=otsu_d(T1,tT1,256)
OTSUT1=create_binary_2(T1,tT1,tOTSUT1)
plt.figure()
ax1=plt.subplot(121)
plt.axis('off')
ax1.set_title('origin')
ax2=plt.subplot(122)
plt.axis('off')
ax1.imshow(T1)
ax2.set_title('gray matter segmentation')

ax2.imshow(OTSUT1)

tT1_v2=otsu(T1_v2)
tT1_v2_2=otsu_d(T1_v2,tT1_v2,256)
OTSUtT1_v2=create_binary_2(T1_v2,tT1_v2,tT1_v2_2)
plt.figure()
ax1=plt.subplot(121)
plt.axis('off')
ax1.set_title('origin')
ax1.imshow(T1_v2)
ax2=plt.subplot(122)
plt.axis('off')
ax2.set_title('gray matter segmentation')
ax2.imshow(OTSUtT1_v2)

tT1_v3=otsu(T1_v3)
tT1_v3_2=otsu_d(T1_v3,tT1_v3,256)
OTSUtT1_v3=create_binary_2(T1_v3,tT1_v3,tT1_v3_2)
plt.figure()
ax1=plt.subplot(121)
plt.axis('off')
ax1.set_title('origin')
ax1.imshow(T1_v3)
ax2=plt.subplot(122)
plt.axis('off')
ax2.set_title('gray matter segmentation')
ax2.imshow(OTSUtT1_v3)

tFlair=otsu(Flair)
OTSUFlair=create_binary(Flair,tFlair)
tOTSUFalir=otsu_d(Flair,tFlair,256)
OTSUFlair1=create_binary(Flair,tOTSUFalir)
plt.figure()
ax1=plt.subplot(121)
plt.axis('off')
ax1.set_title('origin')
ax1.imshow(Flair)
ax2=plt.subplot(122)
plt.axis('off')
ax2.set_title('gray matter segmentation')

ax2.imshow(OTSUFlair1)

tT2=otsu(T2)
tT22=otsu_d(T2,tT2,256)
tT23=otsu_d(T2,tT2,tT22)
OTSUtT2=create_binary_2(T2,tT23,tT22)
plt.figure()
ax1=plt.subplot(121)
plt.axis('off')
ax1.set_title('origin')
ax1.imshow(T2)
ax2=plt.subplot(122)
plt.axis('off')
ax2.set_title('gray matter segmentation')
ax2.imshow(OTSUtT2)


####TOF
import nibabel 
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, hessian
img_data=nibabel.loadsave.load('I:/image/tof_brain.nii')
img=img_data.get_fdata()
#img2=frangi(img,black_ridges=False)
img[np.where(img<245)]=0
img[np.where(img>=245)]=1
save_file=nibabel.Nifti1Image(img, img_data.affine)
nibabel.save(save_file,'I:/image/tofseg_final.nii')
####SWI
import nibabel 
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, hessian
img_data=nibabel.loadsave.load('I:/image/a4/swi_brain.nii')
img=img_data.get_fdata()
img=frangi(img)
save_file=nibabel.Nifti1Image(img, img_data.affine)
nibabel.save(save_file,'I:/image/a4/swi_frangi.nii')
img[np.where(img<0.07)]=0
img[np.where(img>=0.07)]=1
save_file=nibabel.Nifti1Image(img, img_data.affine)
nibabel.save(save_file,'I:/image/a4/swibrainseg_final.nii')





