# -*- coding: utf-8 -*-
"""
Image pre-processing toolkit.
Implementaion of image pre-processing and and image feature extraciton 
functions.

Created on Sun Feb 22 12:22:36 2020

@author Matt Wright

"""
import numpy as np
import scipy.signal as sig
import cv2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


# Applies a HOG kernel to the given matrix. Fust be a single-channel image
def sobel_transform(img):
    # Define the Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Apply kernel
    G_x = sig.convolve2d(img, kernel_x, mode='same') 
    G_y = sig.convolve2d(img, kernel_y, mode='same') 
    
    # Return x abd y sum
    return(G_x + G_y)
    
# Creates instance of HOG descriptor
def declare_HOG():
    winSize = (128,128)
    blockSize = (64,64)
    blockStride = (32,32)
    cellSize = (32,32)
    nbins=9
    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# Determines coordinates of 100 strong corners in the given picture
def SIFT_transform(img):
    corners = cv2.goodFeaturesToTrack(img,100,0.01,10)
    corners = np.int0(corners)
    return(corners)

# Normalizes pixel values and shapes image into approptiate dimensions form model
def prep_data(imgs): 
    data = np.array(imgs)
    data = data.astype('float32') / 255.
    data = np.reshape(data, (len(data), 240, 256, 1))
    return data

# Builds a new autoencoder model. Returns entire autoencoder as well as the 
# the encoding half. Model still needs to be trained
def build_autoencoder():
    input_img = Input(shape=(240, 256, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Complete model - from input image to reconstructed image
    autoencoder = Model(input_img, decoded)
    
    # Half model - outputs reduced image
    encoder = Model(input_img, encoded)
    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    return autoencoder, encoder
    
