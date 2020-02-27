# -*- coding: utf-8 -*-
"""
Second stage or image preprocessing.
Uses grey and sobel filtered images to train an autoencoder and encode the
images into a 30 by 256 matrix. Saves data in new dataframe and saves models.

Created on Tue Feb 24 02:26:49 2020

@author: Matt Wright
"""
import pandas as pd
from feature_extraction_toolkit import prep_data, build_autoencoder
from keras.callbacks import TensorBoard


data = pd.read_pickle('processed_data_stage1.pkl')
addresses = list(data['Addresses']) # want addresses and hog still
hog = list(data['HOG'])


# Begin with Gray scale:
imgs = prep_data(list(data['Gray']))

autoencoder_grey, encoder_grey = build_autoencoder()

autoencoder_grey.fit(imgs, imgs,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_split=0.1,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder_grey.save('saved_autoencoders/grey_autoencoder')
encoder_grey.save('saved_autoencoders/grey_encoder')

encoded_imgs_gray = encoder_grey.predict(imgs)
encoded_imgs_gray = [a.reshape(30, 32*8).T for a in encoded_imgs_gray]


# Now sobel pictures
imgs = prep_data(list(data['Sobel']))

autoencoder_sobel, encoder_sobel = build_autoencoder()

autoencoder_sobel.fit(imgs, imgs,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_split=0.1,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder_sobel.save('saved_autoencoders/sobel_autoencoder')
encoder_sobel.save('saved_autoencoders/sobel_encoder')

encoded_imgs_sobel = encoder_sobel.predict(imgs)
encoded_imgs_sobel = [a.reshape(30, 32*8).T for a in encoded_imgs_sobel]

# Save encoded images
encoded_data_df = pd.DataFrame({'Addresses' : addresses, 
                                 'Gray'     : encoded_imgs_gray, 
                                 'Sobel'    : encoded_imgs_sobel,
                                 'HOG'      : hog})

encoded_data_df.to_pickle('processed_data_stage2.pkl')
