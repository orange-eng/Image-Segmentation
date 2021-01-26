from __future__ import print_function,division

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import *
from keras.models import *
import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import sys
import os
apath = os.path.abspath(os.path.dirname(sys.argv[0]))
import cv2 as cv 



def get_encoder(input_height=416,input_width=416):
    img_Input = Input((input_height,input_width,3))
    
    x = Conv2D(64,kernel_size=3,activation='relu',strides=1,padding='SAME')(img_Input)
    x = Conv2D(64,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    # 224*224*64
    x = MaxPool2D(pool_size=(2,2),strides=2)(x)
    f1 = x

    x = Conv2D(128,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    x = Conv2D(128,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    # 112*112*128
    x = MaxPool2D(pool_size=(2,2),strides=2)(x)
    f2 = x
    
    x = Conv2D(256,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    x = Conv2D(256,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    x = Conv2D(256,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    # 56*56*256
    x = MaxPool2D(pool_size=(2,2),strides=2)(x)
    f3 = x

    x = Conv2D(512,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    x = Conv2D(512,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    x = Conv2D(512,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    # 28*28*512
    x = MaxPool2D(pool_size=(2,2),strides=2)(x)
    f4 = x
    
    x = Conv2D(512,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    x = Conv2D(512,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    x = Conv2D(512,kernel_size=3,activation='relu',strides=1,padding='SAME')(x)
    # 14*14*512
    f5 = x
    img_Output = x
    return img_Input,[f1,f2,f3,f4,f5]

def get_decoder(decoder_input,n_classes):
    x = decoder_input

    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(512,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)    

    x = UpSampling2D(size=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(128,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)   

    x = UpSampling2D(size=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(64,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)

    x = Conv2D(n_classes,(3,3),padding='SAME')(x)

    return x

def SegNet(n_classes,input_height=224,input_width=224,encoder_level=4):
    img_input,levels = get_encoder(input_height=input_height,input_width=input_width)
    #获取压缩4次时的结果，赋值为feat
    feat = levels[encoder_level]

    x = get_decoder(decoder_input=feat,n_classes=n_classes)

    x = Reshape((int(input_height/2)*int(input_width/2),-1))(x)
    #x = Softmax()(x)

    model = Model(img_input,x)
    
    return model
