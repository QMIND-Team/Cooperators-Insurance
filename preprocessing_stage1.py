# -*- coding: utf-8 -*-
"""
Initial stage of image preprocessing.
Loads images, resizes, and stores coloured, grayscale, gradient vector (sobel),
and histogram of oreinted gradients (HOG) repredsentations of each image.

Created on Sun Feb 23 19:13:55 2020

@author: Matt Wright
"""
import cv2
import os
import pandas as pd
from feature_extraction_toolkit import declare_HOG, sobel_transform


img_dir = 'C:\\Users\\mattr\\QMIND\\Cooperators-Insurance\\StreetViews_V2'
img_names = os.listdir(img_dir)
imgs_orig = []      # For original, coloured streetview
imgs_gray = []      # Gray-scale images
imgs_sobel = []     # Contains sobel filtered images
imgs_HOG  = []      # Streetviews after HOG transformation

HOG_transformer = declare_HOG()

for img in img_names:
    pic = cv2.imread(img_dir + '\\' + img)              # Load image
    pic = cv2.resize(pic, (0,0), fx=0.4, fy=0.4)        # Resize image
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)        # Turn to grayspace
    sobel = sobel_transform(gray)                       # Apply sobel filter
    hog = HOG_transformer.compute(gray)                 # Perform HOG
    
    # Add each representaion to an array
    imgs_orig.append(pic)
    imgs_gray.append(gray)
    imgs_sobel.append(sobel)
    imgs_HOG.append(hog)

# Create dataframe for each feature    
data_df = pd.DataFrame({'Addresses' : img_names, 
                         'Color'    : imgs_orig, 
                         'Gray'     : imgs_gray, 
                         'Sobel'    : imgs_sobel,
                         'HOG'      : imgs_HOG  })

# Save the data frame
data_df.to_pickle('processed_data_stage1.pkl')

