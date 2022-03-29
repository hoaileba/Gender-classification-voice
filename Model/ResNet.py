import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import librosa.display
import numpy, scipy, IPython.display as ipd, sklearn
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D,\
                         MaxPooling2D, GlobalMaxPooling2D,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import IPython.display as ipd
import librosa
import glob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import tensorflow.keras.backend as K
import logging
from tensorflow.keras.initializers import glorot_uniform
import os 
import tensorflow_addons as tfa
def identity_block(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2 = filters

    X_shortcut = X
    # print(X.shape)
    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a' )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tfa.activations.gelu(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b' )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tfa.activations.gelu(X)
    
    # X = Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=conv_name_base + '2c' )(X)
    # X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # print(X_shortcut.shape, X.shape)
    X = Add()([X, X_shortcut])# SKIP Connection
    X = tfa.activations.gelu(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2 = filters

    X_shortcut = X
    # print(X_shortcut.shape, X.shape)
    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a' )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b' )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '1' )(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    # print(X_shortcut.shape, X.shape)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet(input_audio=(224, 224, 3), training = True):

    X_input = Input(input_audio)
    X = Conv2D(64, (3, 3), strides=(2, 2), name='conv1')(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tfa.activations.gelu(X)

    X = Conv2D(filters=64, kernel_size=(3,3), strides=2, padding='same', name='a' )(X)
    X = BatchNormalization(axis=3, name='2ba')(X)
    X = tfa.activations.gelu(X)

    X = identity_block(X, f = 3, filters = [64,64], stage = 2, block = 'b')

    X = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same', name='b' )(X)
    X = BatchNormalization(axis=3, name= '2bb')(X)
    X = tfa.activations.gelu(X)    
    X = identity_block(X, f = 3, filters = [128,128], stage=3, block='b')
    
    X = GlobalAveragePooling2D()(X)
    # print("avg shape: ", X.shape)
    X = Flatten()(X)
    # print("Flatten: ",X.shape)
    # X = tf.keras.layers.Concatenate(axis = 1) ([X,X_gender_input])
    
    X = Dense(512,activation = "relu")(X)
    # if training == True:
    #     X = Dropout(0.5)(X)
    X = Dense(256,activation = "relu")(X)
    # if training == True:
    #     X = Dropout(0.5)(X)
    X = Dense(2,activation = "softmax")(X)
    model = Model(inputs=[X_input], outputs=X, name='ResNet18')

    return model
model  = ResNet(input_audio = (40,16,1))
# model.summary()
