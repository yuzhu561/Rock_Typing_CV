#!/usr/bin/env python3
import numpy as np
import random
import math
from scipy import signal
import imageio
from skimage import segmentation as seg
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
from collections import Counter
from scipy import signal, ndimage
from skimage import morphology as mor
from skimage.segmentation import chan_vese
import time


#=====================================Filters==============================
def Boundary_Extension_2D(Image, Extension):
    Image_size=Image.shape
    X=Image_size[0]
    Y=Image_size[1]
    Extended_Image=np.zeros((X+2*Extension, Y+2*Extension), dtype=Image.dtype)
    Extended_Image[Extension: Extension+X, Extension: Extension+Y]=Image
    Left_Boundary=Image[:, 1:Extension+1]
    Left_Boundary=np.fliplr(Left_Boundary)
    Extended_Image[Extension:X+Extension, 0:Extension]=Left_Boundary
    Right_Boundary=Image[:, Y-Extension-1:Y-1]
    Right_Boundary=np.fliplr(Right_Boundary)
    Extended_Image[Extension:X+Extension, Extension+Y:Y+2*Extension]=Right_Boundary
    Up_Boundary=Extended_Image[Extension+1:2*Extension+1, :]
    Up_Boundary=np.flipud(Up_Boundary)
    Extended_Image[0:Extension, :]=Up_Boundary
    Down_Boundary=Extended_Image[X-1:Extension+X-1, :]
    Down_Boundary=np.flipud(Down_Boundary)
    Extended_Image[Extension+X:X+2*Extension, :]=Down_Boundary
    return Extended_Image


def Average_Filter(img, windowsize):
    img.astype('f4')
    half_win=int(windowsize/2)
    if len(img.shape)==2:
        nu=(2*half_win+1)*(2*half_win+1)
        kernel=np.ones((2*half_win+1, 2*half_win+1), dtype='f8')  
        kernel=kernel*(1.0/nu)
        image=signal.convolve2d(img, kernel, boundary='symm', mode='same')
    if len(img.shape)==3:
        nu=(2*half_win+1)*(2*half_win+1)*(2*half_win+1)
        kernel=np.ones((2*half_win+1, 2*half_win+1, 2*half_win+1), dtype='f8')  
        kernel=kernel*(1.0/nu)
        img_extend=Boundary_Extension_3D(img, half_win) 
        img_extend_fft=np.fft.fftn(img_extend)  
        kernel_extend=np.zeros(img_extend.shape)  
        kernel_extend[0:2*half_win+1, 0:2*half_win+1, 0:2*half_win+1]=kernel
        kernel_extend_fft=np.fft.fftn(kernel_extend) 
        filtered=np.fft.ifftn(img_extend_fft*kernel_extend_fft)
        image=np.real(filtered[2*half_win:filtered.shape[0], 2*half_win:filtered.shape[1], 2*half_win:filtered.shape[2]])
    return image    


def Local_Homogeneity_Filter(img):
    filterSize = 2
    img1=np.ones(img.shape, dtype='u1')
    indx=np.flatnonzero(img<1)
    img1.flat[indx]=0

    arNew = img1
    #print(ar)
    homg = 0.0
    homgIndex = 0
    dem1=img.shape[0]
    dem2=img.shape[1]
    for i in range(dem1):
        for j in range(dem2):
            for k in range(max(0,i-int(filterSize/2)), min(i+int(filterSize/2)+1,dem1)):
                for l in range(max(0,j-int(filterSize/2)), min(j+int(filterSize/2)+1,dem2)):
                    if (img1[k][l] == img1[i][j]):
                        homg += 1
            homg = homg / (filterSize+1)*(filterSize+1);
            if (homg > 4):
                arNew[i,j] = 0
            else:
                arNew[i,j] = 255
            homg = 0    

    '''            
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('1')
    ax[1].imshow(arNew, cmap=plt.cm.gray)
    ax[1].set_title('2')
    ax[2].imshow(arNew, cmap=plt.cm.gray)
    ax[2].set_title('2')
    plt.show() 
    '''
    return arNew


def Save_as_png(path, data, filename):
    dimension=data.shape
    minvalue=np.min(data)
    maxvalue=np.max(data)
    data=255*((data-minvalue)/(maxvalue-minvalue))
    data.astype('u1')
    if len(dimension)==2:
        if filename=='':
            imageio.imwrite(path+'/'+'new_image.png', data)
        else:
            imageio.imwrite(path+'/'+filename, data)
    if len(dimension)==3:
        z=dimension.shape[0]
        length=len(str(z))
        for i in rang(z):
            k=str(i)
            k1=k.zfill(length+1)
            imageio.imwrite(path+'/'+filename+'_'+k1+'.png', data[i, :, :])            
       
def Save_as_tiff(path, data, filename):

    if data.dtype!='uint8':
        data.astype('f4')
        data=255*(data-np.min(data))/(np.max(data)-np.min(data))
        data=data.astype('u1') 
        print(data.dtype, np.max(data), np.min(data))       
    dimension=data.shape
    if len(dimension)==2:
        if filename=='':
            imageio.imwrite(path+'/'+'new_image.tiff', data)
        else:
            imageio.imwrite(path+'/'+filename+'.tiff', data)
    if len(dimension)==3:
        z=dimension[0]
        length=len(str(z))
        for i in range(z):
            k=str(i)
            k1=k.zfill(length+1)
            imageio.imwrite(path+'/'+filename+'_'+k1+'.tiff', data[i, :, :]) 


if __name__=='__main__':

    window=50 # window size can be adjusted 
    img=Image.open('/original_image_path/CR1.jpg').convert('L')
    img=np.array(img)
    X=img.shape[0]
    Y=img.shape[1]

    extend_window=int(window/2)
    img_e=Boundary_Extension_2D(img, extend_window)
    
    img_f=np.array(img_e>127)+0 # if the import image is not a binary image with 0 and 1

    img_f_t=Local_Homogeneity_Filter(img_f)

    #Save_as_tiff('save_path', img_filtered, 'CR2_LHF')

    #Save_as_tiff('save_path', img_f, 'CR2_LHF_Average')

    #Save_as_tiff('save_path', img_f_t, 'CR2_LHF_Average_threshold')

    img_f_t_average=Average_Filter(img_f_t, 50)
    #Save_as_tiff('save_path', img_f_t_average, 'CR2_LHF_Average_threshold_average')

    img_f_t_average_cv=chan_vese(img_f_t_average, mu=0.01, lambda1=1, lambda2=1, tol=1e-3, max_iter=200, dt=0.5, init_level_set="checkerboard", extended_output=True)
    img_f_t_average_cv_1=img_f_t_average_cv[0]
    img_f_t_average_cv_1=img_f_t_average_cv_1[extend_window:extend_window+X+1, extend_window:extend_window+Y+1]
    #Save_as_tiff('save_path', img_f_t_average_cv_1, 'CR2_LHF_Average_threshold_average_CV')

    img_f_t_average_cv_rm=mor.remove_small_objects(img_f_t_average_cv_1, min_size=600, connectivity=1, in_place=False)
    #Save_as_tiff('save_path', img_f_t_average_cv_rm, 'CR2_LHF_Average_threshold_average')
    

    fig, axes = plt.subplots(2, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('original image')
    ax[1].imshow(img_f, cmap=plt.cm.gray)
    ax[1].set_title('binary image')
    ax[2].imshow(img_f_t, cmap=plt.cm.gray)
    ax[2].set_title('LHF')
    ax[3].imshow(img_f_t_average, cmap=plt.cm.gray)
    ax[3].set_title('average Filtering')
    ax[4].imshow(img_f_t_average_cv[0], cmap=plt.cm.gray)
    ax[4].set_title('chan-Vese rock typing')
    ax[5].imshow(img_f_t_average_cv_rm, cmap=plt.cm.gray)
    ax[5].set_title('remove small objects')
    plt.show() 

    



















